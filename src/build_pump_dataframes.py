"""
Module takes in a raw excel file and returns a pandas dataframe ready for plotting.
The modules builds numerous additional columns used for plotting, including generating
pump curves for various number of parallel pumps (number of pump defined in the config file),
and allowable and preferable operating ranges  for various speeds.
"""
# pylint: disable=C0103

import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Type, Union

import pandas as pd  # type: ignore
from openpyxl import Workbook  # type: ignore

from helper_funcs import poly_fit, read_config_file

pd.options.mode.chained_assignment = None  # default='warn'


def parse_excel_input(file_location: str) -> pd.DataFrame:
    """
    returns a pandas dataframe from an inputted excel file
    """
    df = pd.read_excel(file_location)
    df = df.rename(
        columns=lambda x: x.strip()
    )  # removing any whitespace from column headers
    return df


def create_pump_dataframe(raw_excel_df: pd.DataFrame, num_pumps: int) -> pd.DataFrame:
    """
    Uses the provided excel sheet to
    Create a dataframe consisting of the pump flow,
    head, bep flow, and bep head for the number
    of pumps specified.

    Renaming columns to include the speed (100%) removes a lot of if/else conditionals
    in other functions
    """
    rename_dict = {
        "pump_head": "pump_head_100",
        "pump_flow": "pump_flow_100",
        "bep_flow": "bep_flow_100",
        "bep_head": "bep_head_100",
    }
    _pump_df = raw_excel_df[["pump_head", "bep_head"]]
    _pump_df["pump_flow"] = raw_excel_df["pump_flow"] * num_pumps
    _pump_df["bep_flow"] = raw_excel_df["bep_flow"] * num_pumps
    _pump_df = _pump_df.rename(columns=rename_dict)
    _pump_df = _pump_df.dropna(axis="index", how="all")
    return _pump_df


def create_pump_df_dict(raw_excel_df: pd.DataFrame, num_pumps: int) -> Dict:
    """
    for each pump in range 1 to num_pumps,
    creates a dictionary and fills it with a dataframe specific to that
    number of parallel pumps
    """
    pump_dict = {}
    for pump in range(1, num_pumps + 1):
        pump_dict[pump] = create_pump_dataframe(raw_excel_df, pump)

    return pump_dict


def create_system_curve_df(raw_excel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads in the raw excel dataframe and returns a dataframe with all columns
    that are not related to the pump curves.
    """
    cols = list(raw_excel_df.columns)
    sys_cols = [word for word in cols if 'pump' not in word]
    sys_cols = [word for word in sys_cols if 'bep' not in word]
    sys_df = raw_excel_df[sys_cols].dropna(axis = 0,how='any')
    return sys_df


def affinity(percent_speed: Union[int, float]) -> Tuple:
    """calculates ratio for head/flow for given speed in %"""
    flow_multiplier = percent_speed / 100
    head_multiplier = (percent_speed / 100) ** 2
    return flow_multiplier, head_multiplier


def apply_speed(
    pump_df: pd.DataFrame, percent_speed: Union[int, float]
) -> pd.DataFrame:
    """
    Applied affinity laws to a given pump dataframe for a given percent speed.
    The function will return the same dataframe with additional columns
    for flow, head, bep flow, and bep head for the given speed.

    """
    flow_multiplier = affinity(percent_speed)[0]
    head_multiplier = affinity(percent_speed)[1]

    pump_df[f"flow_{percent_speed}"] = pump_df["pump_flow_100"] * flow_multiplier
    pump_df[f"head_{percent_speed}"] = pump_df["pump_head_100"] * head_multiplier
    pump_df[f"bep_flow_{percent_speed}"] = pump_df["bep_flow_100"] * flow_multiplier
    pump_df[f"bep_head_{percent_speed}"] = pump_df["bep_head_100"] * head_multiplier
    return pump_df


def apply_aor_and_por(
    pump_df: pd.DataFrame,
    speed: Union[int, float],
    aor_range: Tuple[float, float] = (0.5, 1.25),
    por_range: Tuple[float, float] = (0.75, 1.20),
    degree: int = 2,
) -> Union[pd.DataFrame, Type[Exception]]:
    """
    Calculates the allowable operating range (AOR) for a given pump dataframe.
    The AOR flow is calculated by applying the arguemnts provided for the aor_range to the
    bep_flow column in the pump df.

    The AOR head is calculated by determining the pump curve equation and plugging this
    AOR flow into it.

    The function returns the same df with additional columns for aor_flow_upper, aor_flow_lower,
    aor_head_upper, and aor_head_lower.
    the upper/lower values corresponds to the greater/lesser aor_range argument.
    i.e it corresponds to a higher/lower *flow* not higher head.

    Note the aor_range argument should be provided as ratios of the bep flow. For example,
    the default values (0.5,1.25) represent an AOR range of 50% to 125% of the bep_flow.
    """
    if any(value >= 10 for value in aor_range) or any(
        value >= 10 for value in por_range
    ):
        raise NotImplementedError(
            """\nRanges expressed as percentages are not supported,
            \nplease provide the range as ratios, not percentages"""
        )

    pump_df[f"aor_upper_flow_{speed}"] = pump_df[f"bep_flow_{speed}"] * aor_range[1]
    pump_df[f"aor_lower_flow_{speed}"] = pump_df[f"bep_flow_{speed}"] * aor_range[0]
    pump_df[f"por_upper_flow_{speed}"] = pump_df[f"bep_flow_{speed}"] * por_range[1]
    pump_df[f"por_lower_flow_{speed}"] = pump_df[f"bep_flow_{speed}"] * por_range[0]
    flow_values = pump_df[f"flow_{speed}"]
    head_values = pump_df[f"head_{speed}"]

    pump_df[f"aor_upper_head_{speed}"] = poly_fit(
        x=flow_values,
        y=head_values,
        x_new=pump_df[f"aor_upper_flow_{speed}"],
        deg=degree,
    )
    pump_df[f"aor_lower_head_{speed}"] = poly_fit(
        x=flow_values,
        y=head_values,
        x_new=pump_df[f"aor_lower_flow_{speed}"],
        deg=degree,
    )
    pump_df[f"por_upper_head_{speed}"] = poly_fit(
        x=flow_values,
        y=head_values,
        x_new=pump_df[f"por_upper_flow_{speed}"],
        deg=degree,
    )
    pump_df[f"por_lower_head_{speed}"] = poly_fit(
        x=flow_values,
        y=head_values,
        x_new=pump_df[f"por_lower_flow_{speed}"],
        deg=degree,
    )
    return pump_df


def get_aor_por_limits(pump_df: pd.DataFrame, speed: Union[float, int]) -> Dict:
    """
    For a given pump dataframe and speed grabs the minimum and maximum
    AOR and POR values for that speed. This is required for plotting the
    AOR/POR areas.
    """
    aor_por_names = [
        "aor_upper_flow",
        "aor_lower_flow",
        "aor_upper_head",
        "aor_lower_head",
        "por_upper_flow",
        "por_lower_flow",
        "por_upper_head",
        "por_lower_head",
    ]

    aor_por_limit_dict = {}
    for name in aor_por_names:
        aor_por_limit_dict[name] = pump_df[f"{name}_{speed}"][0]
    return aor_por_limit_dict


def apply_aor_por_limits(pump_df: pd.DataFrame, speed_range: Iterable) -> pd.DataFrame:
    """
    applies the aor/por limit function to a range of speeds and appends the
    full range of speed limits to the pump dataframe
    """
    aor_por_limit_col_names = list(
        get_aor_por_limits(pump_df, speed_range[0]).keys()  # type: ignore
    )  # type: ignore
    # grabbing the aor_por names defined in the get_aor_por_limits func
    aor_por_limit_df = pd.DataFrame(columns=aor_por_limit_col_names)
    for col_name in aor_por_limit_col_names:
        temp_list = []
        for speed in speed_range:
            temp_list.append(get_aor_por_limits(pump_df, speed)[col_name])
        aor_por_limit_df[col_name] = temp_list

    pump_df = pd.concat([pump_df, aor_por_limit_df], axis=1)
    del aor_por_limit_df
    return pump_df


def build_pump_df(initial_pump_df: dict, pump_config_dict: dict) -> pd.DataFrame:
    """
    builds the full pump df for a given raw pump df with only 100 flow, head,
    and best efficiency points.
    Returns the fully constructed df for all speeds

    pump_flow_df should be a df from the create_pump_df_dict function
    that contains only the information from the raw excel file
    """

    for speed in pump_config_dict["data_speeds"]:
        full_pump_df = apply_speed(initial_pump_df, speed)
        full_pump_df = apply_aor_and_por(
            full_pump_df,
            speed,
            aor_range=pump_config_dict["aor_range"],
            por_range=pump_config_dict["por_range"],
            degree=pump_config_dict["aor_por_fit_degree"],
        )
    full_pump_df = apply_aor_por_limits(full_pump_df, pump_config_dict["data_speeds"])
    return full_pump_df


def save_df(
    df: pd.DataFrame,
    filename: str,
    sheet_name: Optional[str],
    save_folder: Union[str, Path] = Path("outputs"),
):
    """
    saves a dataframe to an excel file
    uses openpyxl to create the excel file if it doesnt exist and can save multiple dataframe
    to different sheets
    """

    if Path(filename).suffix != ".xlsx":
        filename = filename + ".xlsx"
    file_dir = Path(save_folder, filename)
    if not file_dir.exists():
        wb = Workbook()
        wb.save(file_dir)
    if sheet_name is None:
        sheet_name = "sheet1"
    with pd.ExcelWriter(
        file_dir, mode="a", engine="openpyxl", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name=sheet_name)
    print(f"df saved to {file_dir}, sheet: {sheet_name}")


def save_pickle(object, file_dir: str):
    """
    saves object to pickle in a given file location
    """
    with open(file_dir, "wb") as f:
        pickle.dump(object, f)


def main():
    """
    main function that calls all other funcs
    """
    config_dir = Path("config.yml")
    pump_config_dict = read_config_file(config_dir)["pump_config"]
    excel_dir = Path("inputs", pump_config_dict["excel_filename"])
    excel_df = parse_excel_input(excel_dir)
    sys_df = create_system_curve_df(excel_df)
    pump_dict = create_pump_df_dict(
        raw_excel_df=excel_df, num_pumps=pump_config_dict["num_pumps"]
    )
    pump_df_list = [build_pump_df(df, pump_config_dict) for df in pump_dict.values()]
    save_config_dict = read_config_file(config_dir)["save_config"]
    if save_config_dict["save_dataframe_to_excel"]:
        for i, df in enumerate(pump_df_list):
            save_df(
                df,
                filename=save_config_dict["excel_save_filename"],
                sheet_name=f"{i+1}_pump(s)",
            )

    save_pickle(sys_df, Path("outputs", "sys_curve_df.pkl"))
    save_pickle(pump_df_list, Path("outputs", "pump_df_list.pkl"))


if __name__ == "__main__":
    main()
