"""PySpark helpers to clean and unify the iFood offer-optimization case raw data.

Pipeline: raw JSONs -> cleaned/flattened tables -> unified event log -> opportunity table
(grain: one row per "offer received" instance, with viewed/completed flags matched inside
that instance's own time window).
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# time_since_test_start values at which offers were sent, confirmed by direct data inspection.
ROUND_DAYS = [0, 7, 14, 17, 21, 24]


def clean_profile(profile: DataFrame) -> DataFrame:
    """Treat age==118 as missing, normalize gender to F/M/O/Unknown, parse registration date."""
    df = profile.withColumn(
        "age",
        F.when(F.col("age") == 118, F.lit(None).cast("int")).otherwise(F.col("age").cast("int")),
    )
    df = df.withColumn(
        "gender",
        F.when(F.upper(F.trim(F.col("gender"))).isin("F", "M", "O"), F.upper(F.trim(F.col("gender"))))
        .otherwise(F.lit("Unknown")),
    )
    df = df.withColumn(
        "registered_on_date", F.to_date(F.col("registered_on").cast("string"), "yyyyMMdd")
    )
    return df.select("id", "age", "gender", "credit_card_limit", "registered_on_date")


def flatten_transactions(transactions: DataFrame) -> DataFrame:
    """Unify the two inconsistent offer-id keys inside the nested `value` struct and cast types."""
    return (
        transactions.withColumn(
            "offer_id", F.coalesce(F.col("value.offer_id"), F.col("value.`offer id`"))
        )
        .withColumn("amount", F.col("value.amount").cast("double"))
        .withColumn("reward", F.col("value.reward").cast("double"))
        .withColumn("time_since_test_start", F.col("time_since_test_start").cast("double"))
        .drop("value")
    )


def build_joined_event_log(
    transactions_flat: DataFrame, profile_clean: DataFrame, offers: DataFrame
) -> DataFrame:
    """One row per event, enriched with customer profile and offer attributes."""
    t = transactions_flat.alias("t")
    p = profile_clean.alias("p")
    o = offers.alias("o")

    return t.join(p, F.col("t.account_id") == F.col("p.id"), "left").join(
        o, F.col("t.offer_id") == F.col("o.id"), "left"
    ).select(
        F.col("t.account_id").alias("customer_id"),
        F.col("t.event").alias("event"),
        F.col("t.time_since_test_start").alias("time_since_test_start"),
        F.col("t.offer_id").alias("offer_id"),
        F.col("t.amount").alias("amount"),
        F.col("t.reward").alias("reward"),
        F.col("p.age").alias("age"),
        F.col("p.gender").alias("gender"),
        F.col("p.credit_card_limit").alias("credit_card_limit"),
        F.col("p.registered_on_date").alias("registered_on_date"),
        F.col("o.offer_type").alias("offer_type"),
        F.col("o.min_value").alias("min_value"),
        F.col("o.duration").cast("double").alias("duration"),
        F.col("o.discount_value").alias("discount_value"),
        F.col("o.channels").alias("channels"),
    )


def build_opportunities(df_joined: DataFrame) -> DataFrame:
    """Build the opportunity table.

    Grain: (customer_id, offer_id, t_received) - one row per instance of an offer being sent,
    even if the same offer was sent to the same customer more than once.

    For each instance, `offer viewed` / `offer completed` events for that (customer, offer_id)
    are matched only if they fall inside that instance's own window
    [t_received, t_received + duration] - this avoids leaking outcomes across repeated
    instances of the same offer to the same customer.

    Known simplification: if the same offer is sent twice to a customer with overlapping
    windows (possible since some rounds are only 3-4 days apart), a single viewed/completed
    event could satisfy both instances' windows and get counted for each. This is rare given
    the round spacing and is accepted as a documented premise.

    Adds `opp_id`, a unique surrogate key per row (safe to join on afterwards - unlike the
    natural key columns, which can contain nulls such as `t_completed`, and NULL <> NULL in
    SQL equi-joins would silently drop rows).
    """
    base = df_joined.select(
        "customer_id",
        "event",
        F.col("time_since_test_start").alias("t"),
        "offer_id",
        "offer_type",
        "duration",
        "age",
        "gender",
        "credit_card_limit",
        "min_value",
        "discount_value",
        "channels",
    )

    received = (
        base.filter(F.col("event") == "offer received")
        .select(
            "customer_id",
            "offer_id",
            "offer_type",
            "duration",
            "age",
            "gender",
            "credit_card_limit",
            "min_value",
            "discount_value",
            "channels",
            F.col("t").alias("t_received"),
        )
        .withColumn("t_expires", F.col("t_received") + F.col("duration"))
        .withColumn("round_day", F.round(F.col("t_received")).cast("int"))
    )

    viewed = base.filter(F.col("event") == "offer viewed").select(
        "customer_id", "offer_id", F.col("t").alias("t_viewed")
    )
    completed = base.filter(F.col("event") == "offer completed").select(
        "customer_id", "offer_id", F.col("t").alias("t_completed")
    )

    received_cols = received.columns

    received_view = (
        received.join(viewed, on=["customer_id", "offer_id"], how="left")
        .withColumn(
            "t_viewed_in_window",
            F.when(
                (F.col("t_viewed") >= F.col("t_received")) & (F.col("t_viewed") <= F.col("t_expires")),
                F.col("t_viewed"),
            ),
        )
        .groupBy(*received_cols)
        .agg(F.min("t_viewed_in_window").alias("t_viewed"))
    )

    received_view_cols = received_view.columns

    opps = (
        received_view.join(completed, on=["customer_id", "offer_id"], how="left")
        .withColumn(
            "t_completed_in_window",
            F.when(
                (F.col("t_completed") >= F.col("t_received")) & (F.col("t_completed") <= F.col("t_expires")),
                F.col("t_completed"),
            ),
        )
        .groupBy(*received_view_cols)
        .agg(F.min("t_completed_in_window").alias("t_completed"))
        .withColumn("viewed", F.when(F.col("t_viewed").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("completed", F.when(F.col("t_completed").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("opp_id", F.monotonically_increasing_id())
    )

    return opps


def add_informational_success(opps: DataFrame, df_joined: DataFrame) -> DataFrame:
    """For informational offers (no 'completed' event exists), flag whether a transaction
    happened inside the window AND after the customer viewed the offer (viewing is required -
    an informational offer can't have "worked" if the customer never saw it).
    """
    tx = df_joined.filter(F.col("event") == "transaction").select(
        "customer_id", F.col("time_since_test_start").alias("t_tx")
    )

    info = opps.filter(F.col("offer_type") == "informational")
    other = opps.filter(F.col("offer_type") != "informational")

    info_tx = (
        info.select("opp_id", "customer_id", "t_viewed", "t_expires")
        .join(tx, on="customer_id", how="left")
        .where(
            F.col("t_viewed").isNotNull()
            & (F.col("t_tx") >= F.col("t_viewed"))
            & (F.col("t_tx") <= F.col("t_expires"))
        )
        .groupBy("opp_id")
        .agg(F.lit(1).alias("informational_success"))
    )

    info_with_flag = info.join(info_tx, on="opp_id", how="left").withColumn(
        "informational_success", F.coalesce(F.col("informational_success"), F.lit(0))
    )

    other_with_flag = other.withColumn("informational_success", F.lit(None).cast("int"))

    return other_with_flag.unionByName(info_with_flag)
