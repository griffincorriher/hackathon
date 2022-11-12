# Import packages
import pandas as pd
import numpy as np


# Import required files
df_submitted = "< Name of your dataframe that contains the result >"
df_total_tires_public = pd.read_csv("./public/Challenge2_total_tires_public.csv")
df_vehicle_public = pd.read_csv("./public/Challenge2_total_tires_public.csv")
df_distance_public = pd.read_csv("./public/Challenge2_distances_public.csv")
df_second_leg_GHG_public = pd.read_csv("./public/Challenge2_second_leg_GHG_public.csv")


# Calculate wape
def simple_wape(y_true, y_pred):
    """Calculates simple wape"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.round(
        abs(y_true - y_pred).sum() / abs(y_true).sum()
        if abs(y_true).sum() != 0
        else np.inf,
        5,
    )


# Calculate total GHG
def total_GHG(
    df_submitted,
    df_total_tires_public,
    df_vehicle_public,
    df_distance_public,
    df_second_leg_GHG_public,
):
    """
    Args:
        df_submitted:             Dataframe submitted by the contestant.
        df_total_tires_public:    Dataframe showing total tires per DC
        df_vehicle_public:        Dataframe showing vehicles available per DC
        df_distance_public:       Dataframe showing distance from DC to various locations
        df_second_leg_GHG_public: Dataframe showing extra GHGs they incur if tires are not dropped to recycler.

    Return:
        Returns total GHG.
    """
    # Making sure their submission is capitalized!
    df_submitted = df_submitted.applymap(lambda s: s.upper() if type(s) == str else s)

    # Part 1: calculate GHG for shipped tires
    # Merge with total tires we have across all DC.
    df = df_submitted.merge(df_total_tires_public, on="dc_name", how="right")

    # Set nulls for 'number_of_tires_shipped' to zero
    df.loc[df["number_of_tires_shipped"].isnull(), "number_of_tires_shipped"] = 0

    # Join the vehicle table
    df = df.merge(df_vehicle_public, on=(["dc_name", "vehicle_name"]), how="left")

    # Join the distance table
    df = df.merge(df_distance_public, on=["dc_name", "destination"], how="left")

    # Join second_leg_additional GHG info table
    df = df.merge(
        df_second_leg_GHG_public,
        left_on=["destination"],
        right_on=["location"],
        how="left",
    )

    # First line for shipped tires and second line is for penalty for leaving tire at Hub or Train
    df["shipped_ghg"] = (
        df["base_ghg_per_mile"] * df["distance"]
        + df["number_of_tires_shipped"]
        * df["distance"]
        * df["extra_ghg_per_tire_per_mile"]
        + df["base_ghg"]
        + df["additional_ghg_per_tire"] * df["number_of_tires_shipped"]
    )
    # If they did not ship anything shipped_GHG is zero
    df.loc[df["shipped_ghg"].isnull(), "shipped_ghg"] = 0

    # Part 2: calculate GHG for unshipped tires
    # Calculate total tires shipped per DC
    df = df.groupby("dc_name")["number_of_tires_shipped"].sum().reset_index()

    # Merge on total tires present in a given DC
    df = df.merge(df_total_tires_public, on=["dc_name"])

    # Find tires remaining in the DC
    df["remaining"] = df["tires"] - df["number_of_tires_shipped"]

    # Merge with second_leg_table
    df = df.merge(
        df_second_leg_GHG_public, left_on=["dc_name"], right_on=["location"], how="left"
    )

    # Calculate unshipped GHG
    df["unshipped_ghg"] = (
        df["base_ghg"] + df["remaining"] * df["additional_ghg_per_tire"]
    )

    # Part 3: Calculate total GHG using Part 1 & Part 2
    return np.round((df["shipped_ghg"].sum() + df["unshipped_ghg"].sum()), 4)
