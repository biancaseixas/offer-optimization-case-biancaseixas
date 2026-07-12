"""Point-in-time feature engineering for the iFood offer-optimization case.

Builds the context vector (customer + offer static features) that feeds the
contextual Thompson Sampling bandit in notebook 2. Consumes the opportunity
table produced by `data_processing.build_opportunities` /
`add_informational_success`, plus the enriched event log (`df_joined`), and
computes historical behavior strictly *before* `t_received` to avoid leakage.
"""

from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

KNOWN_CHANNELS = ["email", "mobile", "web", "social"]


def build_context_features(
    opps_full: DataFrame,
    df_joined: DataFrame,
    lookback_days_short: float = 7.0,
    lookback_days_long: float = 28.0,
) -> Tuple[DataFrame, List[str], List[str]]:
    """Build the context table for the bandit.

    Grain: one row per `opp_id` (same grain as `opps_full`).

    Returns:
        context_df: opps_full's key/target columns + engineered features.
        numerical_cols: numerical feature column names.
        categorical_cols: categorical feature column names.
    """
    base = _base_columns(opps_full)
    offer_feats = _offer_static_features(opps_full)
    channel_feats = _channel_features(opps_full)
    profile_feats = _profile_features(opps_full)
    history_feats = _historical_features(opps_full, df_joined, lookback_days_short, lookback_days_long)
    tenure_feats = _tenure_features(opps_full, df_joined)

    context_df = (
        base.join(offer_feats, on="opp_id", how="left")
        .join(channel_feats, on="opp_id", how="left")
        .join(profile_feats, on="opp_id", how="left")
        .join(history_feats, on="opp_id", how="left")
        .join(tenure_feats, on="opp_id", how="left")
    )

    context_df = _fill_defaults(context_df)
    numerical_cols, categorical_cols = _column_types(context_df)

    return context_df, numerical_cols, categorical_cols


def _base_columns(opps_full: DataFrame) -> DataFrame:
    """`reward` unifies success across offer types for the bandit: `completed` for
    bogo/discount (informational never has a completion event - it is
    structurally always 0 for that type, not a real signal), `informational_success`
    for informational (transaction after view, within the window)."""
    return opps_full.select(
        "opp_id",
        "customer_id",
        "offer_id",
        "offer_type",
        "t_received",
        "round_day",
        "viewed",
        "completed",
        "informational_success",
        F.when(F.col("offer_type") == "informational", F.col("informational_success"))
        .otherwise(F.col("completed"))
        .alias("reward"),
        "discount_value",
        "min_value",
    )


def _offer_static_features(opps_full: DataFrame) -> DataFrame:
    df = opps_full.select(
        "opp_id",
        F.col("duration").cast("double").alias("offer_duration_days"),
        F.col("min_value").cast("double").alias("offer_min_spend"),
        F.col("discount_value").cast("double").alias("offer_discount_value"),
    )
    return df.withColumn(
        "offer_reward_ratio",
        F.when(
            F.col("offer_min_spend").isNotNull() & (F.col("offer_min_spend") > 0),
            F.col("offer_discount_value") / F.col("offer_min_spend"),
        ).otherwise(F.lit(0.0)),
    ).select("opp_id", "offer_duration_days", "offer_reward_ratio")


def _channel_features(opps_full: DataFrame) -> DataFrame:
    df = opps_full.select("opp_id", "channels")
    for ch in KNOWN_CHANNELS:
        df = df.withColumn(
            f"channel_is_{ch}",
            F.when(F.array_contains(F.col("channels"), ch), F.lit(1)).otherwise(F.lit(0)).cast("int"),
        )
    return df.drop("channels")


def _profile_features(opps_full: DataFrame) -> DataFrame:
    """Static customer attributes, plus a flag for incomplete-profile customers.

    Notebook 1's EDA found the ~12.8% of customers with age/gender/limit missing
    convert far less (14.7% vs 41-73%+) - worth a flag, not just imputation.
    """
    return opps_full.select(
        "opp_id",
        F.col("age").cast("double").alias("customer_age"),
        F.coalesce(F.col("gender"), F.lit("Unknown")).alias("customer_gender"),
        F.col("credit_card_limit").cast("double").alias("customer_credit_limit"),
        F.when(F.col("age").isNull(), F.lit(1)).otherwise(F.lit(0)).alias("customer_incomplete_profile"),
    )


def _historical_features(
    opps_full: DataFrame,
    df_joined: DataFrame,
    lookback_days_short: float,
    lookback_days_long: float,
) -> DataFrame:
    """Point-in-time behavioral features, using only events strictly before t_received."""
    anchors = opps_full.select("opp_id", "customer_id", "t_received")

    events = df_joined.select(
        "customer_id",
        "event",
        F.col("time_since_test_start").cast("double").alias("t_event"),
        F.col("amount").cast("double").alias("amount"),
    )

    hist = (
        anchors.join(events, on="customer_id", how="left")
        .where(F.col("t_event") < F.col("t_received"))
    )

    in_short = F.col("t_event") >= (F.col("t_received") - F.lit(lookback_days_short))
    in_long = F.col("t_event") >= (F.col("t_received") - F.lit(lookback_days_long))

    tx_agg = (
        hist.where(F.col("event") == "transaction")
        .groupBy("opp_id")
        .agg(
            F.sum(F.when(in_short, 1).otherwise(0)).alias("tx_count_7d"),
            F.sum(F.when(in_long, 1).otherwise(0)).alias("tx_count_28d"),
            F.count("*").alias("transaction_count_before"),
            F.sum(F.coalesce(F.col("amount"), F.lit(0.0))).alias("total_spend_before"),
            F.avg("amount").alias("avg_ticket_before"),
            F.max("t_event").alias("_last_tx_time"),
        )
    )

    offer_hist = (
        hist.where(F.col("event").isin("offer received", "offer viewed", "offer completed"))
        .groupBy("opp_id")
        .agg(
            F.sum(F.when(F.col("event") == "offer received", 1).otherwise(0)).alias("offers_received_count_before"),
            F.sum(F.when(F.col("event") == "offer viewed", 1).otherwise(0)).alias("offers_viewed_count_before"),
            F.sum(F.when(F.col("event") == "offer completed", 1).otherwise(0)).alias("offers_completed_count_before"),
            F.max(F.when(F.col("event") == "offer viewed", F.col("t_event"))).alias("_last_view_time"),
        )
    )

    out = (
        anchors.select("opp_id", "t_received")
        .join(tx_agg, on="opp_id", how="left")
        .join(offer_hist, on="opp_id", how="left")
    )

    out = out.withColumn(
        "days_since_last_tx",
        F.when(F.col("_last_tx_time").isNotNull(), F.col("t_received") - F.col("_last_tx_time")),
    ).withColumn(
        "days_since_last_offer_view",
        F.when(F.col("_last_view_time").isNotNull(), F.col("t_received") - F.col("_last_view_time")),
    )

    out = out.withColumn(
        "view_rate_before",
        F.when(
            F.col("offers_received_count_before") > 0,
            F.col("offers_viewed_count_before") / F.col("offers_received_count_before"),
        ).otherwise(F.lit(0.0)),
    ).withColumn(
        "completion_rate_before",
        F.when(
            F.col("offers_received_count_before") > 0,
            F.col("offers_completed_count_before") / F.col("offers_received_count_before"),
        ).otherwise(F.lit(0.0)),
    )

    return out.drop("t_received", "_last_tx_time", "_last_view_time")


def _tenure_features(opps_full: DataFrame, df_joined: DataFrame) -> DataFrame:
    """Customer tenure (days) at the moment the offer was received.

    Reference date = max(registered_on_date) + 1 day, derived from the data itself
    (not received_time, so tenure is a stable customer attribute, not something
    that changes across an individual customer's own opportunities).
    """
    reg = df_joined.select("customer_id", "registered_on_date").dropDuplicates(["customer_id"])
    max_reg = reg.select(F.max("registered_on_date").alias("m")).first()["m"]
    ref_date = F.date_add(F.lit(max_reg), 1)

    reg = reg.withColumn("customer_tenure_days", F.datediff(ref_date, F.col("registered_on_date")).cast("double"))

    return (
        opps_full.select("opp_id", "customer_id")
        .join(reg.select("customer_id", "customer_tenure_days"), on="customer_id", how="left")
        .select("opp_id", "customer_tenure_days")
    )


def _fill_defaults(df: DataFrame) -> DataFrame:
    """Fills every remaining numerical null.

    Important: leaving NaNs in the context matrix is not just a "missing value"
    issue here - it corrupts the online Bayesian logistic regression update in
    `bandit.py` (a single NaN in `x` propagates through the Newton step and can
    push an arm's weights to +-inf, after which it deterministically "wins"
    every future round regardless of context). So every numerical column must be
    fully non-null before this table is used to build the bandit's design matrix.
    """
    fill_zero = [
        "tx_count_7d",
        "tx_count_28d",
        "transaction_count_before",
        "total_spend_before",
        "avg_ticket_before",
        "offers_received_count_before",
        "offers_viewed_count_before",
        "offers_completed_count_before",
        "view_rate_before",
        "completion_rate_before",
        "offer_reward_ratio",
    ]
    for c in fill_zero:
        if c in df.columns:
            df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

    if "customer_gender" in df.columns:
        df = df.withColumn("customer_gender", F.coalesce(F.col("customer_gender"), F.lit("Unknown")))

    # customer_age / customer_credit_limit: null only for the ~12.8% incomplete-profile
    # customers, already flagged via customer_incomplete_profile - safe to impute with
    # the median so the column itself stays informative for everyone else.
    for c in ["customer_age", "customer_credit_limit"]:
        if c in df.columns:
            median = df.approxQuantile(c, [0.5], 0.01)[0]
            df = df.withColumn(c, F.coalesce(F.col(c), F.lit(median)))

    # days_since_last_tx / days_since_last_offer_view: null means "no prior event in
    # the observed window" (not "recent") - fill with a sentinel just past the
    # experiment's own span (~29 days) rather than a large outlier, which would
    # otherwise dominate standardization downstream.
    for c in ["days_since_last_tx", "days_since_last_offer_view"]:
        if c in df.columns:
            df = df.withColumn(c, F.coalesce(F.col(c), F.lit(30.0)))

    return df


def _column_types(df: DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {
        "opp_id", "customer_id", "offer_id", "offer_type", "t_received", "round_day",
        "viewed", "completed", "informational_success", "reward", "discount_value", "min_value",
        # Static properties of the offer that was actually LOGGED for this row (duration,
        # reward ratio, channel mix) - NOT customer attributes. Each of the 10 offers has
        # an almost-fixed combination of these, so including them in the shared per-row
        # context vector leaks the logged arm's identity into x: a per-arm model trained
        # only on its own kept rows would learn "converts well when duration/channels look
        # like MY OWN offer" - useless within an arm (zero variance) and misleading across
        # arms (scoring arm B using arm A's static fingerprint). Kept in the table itself
        # for reference/business reporting, just excluded from the bandit's design matrix.
        "offer_duration_days", "offer_reward_ratio",
        "channel_is_email", "channel_is_mobile", "channel_is_web", "channel_is_social",
    }
    numerical_types = {"int", "bigint", "float", "double", "decimal", "short", "byte", "long"}
    categorical_types = {"string", "boolean"}

    numerical_cols: List[str] = []
    categorical_cols: List[str] = []
    for col_name, dtype in df.dtypes:
        if col_name in exclude:
            continue
        if dtype in numerical_types:
            numerical_cols.append(col_name)
        elif dtype in categorical_types:
            categorical_cols.append(col_name)

    return numerical_cols, categorical_cols
