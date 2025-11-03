from pathlib import Path
import json
import pandas as pd
from util.build_report import generate_tables_report
from tqdm import tqdm
import time
saved = True
import os
import polars as pl
from pathlib import Path
from typing import Dict, List, Union, Iterable, Tuple
import pandas as pd
from util.build_report import generate_tables_report
import os
import json
saved = True
report = False

start_time = time.time()

def join_csv_by_keywords(
    folder: Union[str, Path],
    given_list: Union[List[str], Dict[str, Iterable[str]]],
    recursive: bool = False,
    case_insensitive: bool = True,
    return_filemap: bool = False,
    read_kwargs: Dict = None,
) -> Union[Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict[str, List[Path]]]]:
    
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not exist.")

    if isinstance(given_list, dict):
        groups = {str(k): [str(p) for p in v] for k, v in given_list.items()}
    else:
        groups = {str(g): [str(g)] for g in given_list}


    files = list(folder.rglob("*.csv") if recursive else folder.glob("*.csv"))


    def norm(s: str) -> str:
        return s.lower() if case_insensitive else s

    files_by_group: Dict[str, List[Path]] = {g: [] for g in groups}
    for f in files:
        fname = norm(f.name)
        for g, pats in groups.items():
            for pat in pats:
                if norm(pat) in fname:
                    files_by_group[g].append(f)
                    break  


    read_kwargs = read_kwargs or {}
    dfs_by_group: Dict[str, pd.DataFrame] = {}
    for g, flist in files_by_group.items():
        if not flist:  
            dfs_by_group[g] = pd.DataFrame()
            continue
        parts = []
        for f in sorted(flist):
            df = pd.read_csv(f, **read_kwargs)
            parts.append(df)
        dfs_by_group[g] = pd.concat(parts, axis=0, ignore_index=True, sort=False)

    return (dfs_by_group, files_by_group) if return_filemap else dfs_by_group


dfs, filemap = join_csv_by_keywords(
    folder="data/click/",
    given_list={"ASSIGNMENTS": ["ASSIGNMENTS"],
                "DEPARTMENT":["DEPARTMENT"],
                "DISTRICTs":["DISTRICTS"],
                "ENGINEERS":["ENGINEERS"],
                "EQUIPMENT":["EQUIPMENT"],
                "TASK_STATUSES":["TASK_STATUSES"],
                "TASK_TYPES":["TASK_TYPES"],
                "TASKS":["TASKS"]
                },
                
    return_filemap=True
)

assignments     = dfs["ASSIGNMENTS"]
departments     = dfs["DEPARTMENT"]
districts       = dfs["DISTRICTs"]      
engineers       = dfs["ENGINEERS"]
equipment       = dfs["EQUIPMENT"]
task_statuses   = dfs["TASK_STATUSES"]
task_types      = dfs["TASK_TYPES"]
tasks           = dfs["TASKS"]

import re
import pandas as pd
import numpy as np

TIME_COLS = {
    "TIMECREATED","TIMEMODIFIED","EARLYSTART","LATESTART","OPENDATE",
    "APPOINTMENTSTART","APPOINTMENTFINISH","DISPLAYDATE","SEMPRAWORKMGMTMODDATE",
    "SCHEDULEDSTART","SCHEDULEDFINISH","ONSITETIMESTAMP","COMPLETIONTIMESTAMP",
    "METRICDATE","DUEDATEBUFFER","Z_EARLYSTART_DATE","Z_DUE_DATE",
    "Z_SCHEDULEDSTART_DATE","Z_SCHEDULEDFINISH_DATE","Z_TIMECREATED_DATE","Z_COMPLETION_DATE"
}

def _parse_two_formats(s: pd.Series):
    s2 = s.copy()
    if s2.dtype == object:
        s2 = s2.astype("string").str.strip().replace("", pd.NA)
    out = pd.to_datetime(s2, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    mask = out.isna() & s2.notna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(s2[mask], format="%Y-%m-%d", errors="coerce")
    return out

def _is_integerish(x: pd.Series):
    y = pd.to_numeric(x, errors="coerce")
    y = y.dropna()
    if y.empty:
        return True 
    return np.all(np.mod(y, 1) == 0)

_BOOL_STR_TRUE  = {"y","yes","true","t"}
_BOOL_STR_FALSE = {"n","no","false","f"}
def _coerce_to_boolean(s: pd.Series):
    if s.dtype == "boolean":
        return s
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    tmp = s
    if s.dtype == object:
        tmp = s.astype("string").str.strip()
        tmp = tmp.replace("", pd.NA)
        tmp = tmp.str.lower()
        tmp = tmp.replace({**{v: True for v in _BOOL_STR_TRUE},
                           **{v: False for v in _BOOL_STR_FALSE}})
    tmp_num = pd.to_numeric(tmp, errors="ignore")
    if isinstance(tmp_num, pd.Series):
        tmp = tmp_num
    tmp = tmp.map(lambda v: True if v in [1, True] else (False if v in [0, False] else (pd.NA if pd.isna(v) else v)))
    return tmp.astype("boolean")

def propose_dtype_spec(df: pd.DataFrame) -> dict:
    spec = {}
    lower_map = {c.lower(): c for c in df.columns}

    id_like = re.compile(r"(?:^|_)(id|key|nbr|number|seq(uence)?|ref|account|costcenter)$", re.I)
    bool_like = re.compile(r"^(is|has)[A-Z_]|(FLAG)$", re.I)  # is*/has* 或以 FLAG 结尾
    date_like = re.compile(r"(date|timestamp|timecreated|timemodified)$", re.I)

    for col in df.columns:
        clen = col.lower()

        if clen in {c.lower() for c in TIME_COLS} or date_like.search(col):
            spec[col] = "datetime64[ns]"
            continue

        if bool_like.search(col) or col.upper() in [
            "OCRFLAGGING2MAN","CMDELIVEREDFLAG","CMREADFLAG","OCRMACHINEDIGGER",
            "SEMPRASUSPENDFLAG","SEMPRASPECIALEQUIPMENTFLAG","SEMPRAREFERFLAG",
            "SEMPRAPREREQUISITESMET","SEMPRADISPATCHREADY",
            "AMOPTOUT","MTUTRANSFLAG","UPLOADPENDINGFLAG","ISCREWTASK","ISSCHEDULED",
            "INJEOPARDY","PINNED","SEMPRATREE","OCRTREE","OCRHAZMAT","OCRENVIRONMENTAL","TASKNUMBER"
        ]:
            spec[col] = "Int64"
            continue

        if id_like.search(col) or col.upper() in ["W6KEY","CALLID","FUNCTLOCREFNBR",
                                                  "CLICKPROJECTCODE","OUTAGEEVENTID","SEMPRAFACILITYID",
                                                  "SEMPRAUSATICKETNBR","SEMPRAACCOUNTNUMBER","SEMPRAMETERBADGENUMBER",
                                                  "SEMPRAINTERRUPTFLAG","SEMPRACPSEQUENCENUMBER",
                                                  "SEMPRACPFACILITYTYPE","SEMPRACOSTCENTER","Z_TASKKEY_CHAR",
                                                  "SEMPRACPCODE","COMPANY","DEPARTMENT","DISTRICT","REGION","CITY","COUNTRYID",
                                                  "POSTCODE","BUSINESSUNIT","COMPANY","DEPARTMENT"]:
            spec[col] = "string"
            continue
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            spec[col] = "datetime64[ns]"
        elif pd.api.types.is_bool_dtype(s):
            spec[col] = "boolean"
        elif pd.api.types.is_integer_dtype(s):
            spec[col] = "Int64"  
        elif pd.api.types.is_float_dtype(s):
            spec[col] = "Float64"  
        elif pd.api.types.is_numeric_dtype(s):
            spec[col] = "Float64"
        else:
            num = pd.to_numeric(s, errors="coerce")
            num_notna = num.notna().sum()
            s_notna = s.notna().sum()
            if s_notna > 0 and num_notna / max(1, s_notna) > 0.9:
                spec[col] = "Int64" if _is_integerish(s) else "Float64"
            else:
                spec[col] = "string" 

    return spec

def enforce_dtype_spec(df: pd.DataFrame, spec: dict, inplace: bool = True):
    out = df if inplace else df.copy()

    for col, target in spec.items():
        if col not in out.columns:
            continue
        s = out[col]

        try:
            if target == "datetime64[ns]":
                out[col] = _parse_two_formats(s)
            elif target == "boolean":
                out[col] = _coerce_to_boolean(s)
            elif target == "Int64":
                out[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
            elif target == "Float64":
                out[col] = pd.to_numeric(s, errors="coerce").astype("Float64")
            elif target == "string":
                out[col] = s.astype("string")
            else:
                out[col] = s.astype(target)
        except Exception as e:
            print(f"[WARN] column '{col}' -> {target} failed: {e}")

    lower_map = {c.lower(): c for c in out.columns}
    for tcol in TIME_COLS:
        if tcol.lower() in lower_map:
            col = lower_map[tcol.lower()]
            if not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = _parse_two_formats(out[col])

    object_cols = [c for c in out.columns if out[c].dtype == object]
    if object_cols:
        print("\n[WARNING] remain object dtype columns, need human judgement")
        for c in object_cols:
            samp = out[c].dropna().astype(str).unique()[:5]
            print(f"  - {c}: object，样本: {list(samp)}")
    return out


def normalize_task_dtypes(tasks: pd.DataFrame, extra_overrides: dict | None = None, inplace: bool = True):
    spec = propose_dtype_spec(tasks)

    if extra_overrides:
        spec.update(extra_overrides)

    out = enforce_dtype_spec(tasks, spec, inplace=inplace)
    return out, spec

tasks_norm, dtype_spec = normalize_task_dtypes(tasks)

def cleaning_assignments(df):
    DT_COLS = ["TIMECREATED", "TIMEMODIFIED", "STARTTIME", "FINISHTIME"]
    FMT = "%Y-%m-%d %H:%M:%S"

    for c in DT_COLS:
        df[c] = pd.to_datetime(df[c], format=FMT, errors="coerce")
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if obj_cols:
        df[obj_cols] = df[obj_cols].astype("string")
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
        df[obj_cols] = df[obj_cols].replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA, "None": pd.NA, "NULL": pd.NA}
    )
    return df


assignments_norm = cleaning_assignments(assignments)


def cleaning_department(df):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if obj_cols:
        df[obj_cols] = df[obj_cols].astype("string")
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
        df[obj_cols] = df[obj_cols].replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA, "None": pd.NA, "NULL": pd.NA}
    )
    return df
departments_norm = cleaning_department(departments)

def cleaning_districts(df):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if obj_cols:
        df[obj_cols] = df[obj_cols].astype("string")
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
        df[obj_cols] = df[obj_cols].replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA, "None": pd.NA, "NULL": pd.NA}
    )
    return df
districts_norm = cleaning_districts(districts)


TIME_COLS |= {
    "PREFERENCEAPPROVALDATE", "ROSTERAPPROVALDATE", "LOGINTIME", "LOGOUTTIME"
}


ENGINEER_OVERRIDES = {
    "TIMECREATED": "datetime64[ns]",
    "TIMEMODIFIED": "datetime64[ns]",
    "PREFERENCEAPPROVALDATE": "datetime64[ns]",
    "ROSTERAPPROVALDATE": "datetime64[ns]",
    "LOGINTIME": "datetime64[ns]",
    "LOGOUTTIME": "datetime64[ns]",

    "NAME": "string",
    "CITY": "string",
    "COMPANY": "string",
    "CONTRACT": "string",
    "BUSINESSUNIT": "string",
    "DEPARTMENT": "string",
    "CREW": "string",
    "LOCATIONID": "string",
    "MOBILECLIENTSETTINGS": "string",
    "MOBILEWAPCLIENTSETTINGS": "string",
    "RELOCATIONSOURCE": "string",
    "CMMDTNUMBER": "string",  


    "W6KEY": "Int64",
    "REVISION": "Int64",
    "DISTRICT": "Int64",
    "CALENDAR": "Int64",
    "ENGINEERTYPE": "Int64",
    "ACTIVE": "Int64",
    "TRAVELSPEED": "Int64",
    "INTERNAL": "Int64",
    "MOBILECLIENT": "Int64",

    "CONTRACTOR": "Int64",
    "FIXEDTRAVEL": "Int64",
    "IGNOREALLPREFERENCES": "Int64",
    "IGNOREFAIRNESSCALCULATION": "Int64",
    "PREFERENCEAPPROVED": "Int64",
    "ROSTERAPPROVED": "Int64",
    "LUNCHBREAKDURATION": "Int64",
    "HASDYNAMICDATA": "Int64",
    "CREWFOREXTERNALUSE": "Int64",

    "AVAILABILITYFACTOR": "Float64",
    "EFFICIENCY": "Float64",
}


engineers_norm, eng_spec = normalize_task_dtypes(engineers, extra_overrides=ENGINEER_OVERRIDES, inplace=False)


obj_cols = engineers_norm.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    engineers_norm[obj_cols] = (
        engineers_norm[obj_cols]
        .astype("string")
        .apply(lambda s: s.str.strip())
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA, "None": pd.NA, "NULL": pd.NA})
    )


TIME_COLS |= {"TIMECREATED","TIMEMODIFIED","STARTUPDATE","INSTALLDATE","MANUFACTURERDATE"}

EQUIPMENT_OVERRIDES = {
    "TIMECREATED": "datetime64[ns]",
    "TIMEMODIFIED": "datetime64[ns]",
    "STARTUPDATE": "datetime64[ns]",
    "INSTALLDATE": "datetime64[ns]",
    "MANUFACTURERDATE": "datetime64[ns]",

    "EQUIPMENTNBR": "string",
    "CALLID": "string",
    "LEGACYID": "string",
    "ZWORKORDERREF": "string",
    "ZEXTWORKORDERNBR": "string",
    "ZTESTEQUIPID": "string",
    "CATALOGPROFILE": "string",
    "NOTIFTYPE": "string",
    "EQUIDESCR": "string",
    "EQUITYPE": "string",
    "MANUFACTURE": "string",
    "MANUFMODEL": "string",
    "MANUFPARTNBR": "string",
    "MANUFSERNBR": "string",
    "MATERIAL": "string",
    "SERIALNBR": "string",
    "TECHIDNBR": "string",
    "ZATLASNBR": "string",
    "ZATLASPREFIX": "string",
    "ZATLASSUFFIX": "string",
    "ZPLATSHEETNBR": "string",
    "ZTHOMASBROTHERSGUIDE": "string",
    "OBJECTTYPEDESC": "string",
    "INTEQUIID": "string",
    "CITY": "string",
    "COUNTRY": "string",
    "NAME1": "string",
    "NAME2": "string",
    "POSTCODE": "string",
    "STATE": "string",
    "STREET": "string",
    "STREET2": "string",
    "STREETNEW": "string",
    "ACTIONCD": "string",
    "ZWORKORDERREF": "string",

    "W6KEY": "Int64",
    "REVISION": "Int64",
    "NBR": "Int64",
    "UNPLANNEDIND": "Int64",
    "EQFOLLOWUPREQD": "Int64",
    "PLANNEDINSPECTIONIND": "Int64",
    "INTSTATUS": "Int64",
    "TASKTYPE": "Int64",
    "PRIORITYCODE": "Int64",
    "MAINTPLANTYPE": "Int64",

    "INSP_ECR": "Int64", "INSP_MTRR": "Int64", "INSP_MTRD": "Int64", "INSP_MTRT": "Int64",
    "INSP_REG": "Int64", "INSP_VLT": "Int64", "INSP_V": "Int64", "INSP_FILT": "Int64",
    "CO_ECR": "Int64", "CO_MTR": "Int64", "CO_REG": "Int64", "CO_VLT": "Int64",
    "CO_V": "Int64", "CO_FILT": "Int64", "INSP_TPG": "Int64", "CO_TPG": "Int64", "CO_MTU": "Int64",

    "CHARTOLERANCE_LOWER": "Float64",
    "CHARTOLERANCE_UPPER": "Float64",
    "CHARREAD_POINT": "Float64",
    "CHARPREVREAD": "Float64",
    "CHARTOLERANCE_LOWER_IO": "Float64",
    "CHARTOLERANCE_UPPER_IO": "Float64",
    "CHARPREVREAD_IO": "Float64",
    "INTPRESENTMVREAD": "Float64", 
    "W6EQKEY": "Int64",              
}


equipment_norm, eq_spec = normalize_task_dtypes(equipment, extra_overrides=EQUIPMENT_OVERRIDES, inplace=False)


obj_cols = equipment_norm.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    equipment_norm[obj_cols] = (
        equipment_norm[obj_cols]
        .astype("string")
        .apply(lambda s: s.str.strip())
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA, "None": pd.NA, "NULL": pd.NA})
    )


def cleaning_task_statuses(df):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if obj_cols:
        df[obj_cols] = df[obj_cols].astype("string")
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
        df[obj_cols] = df[obj_cols].replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA, "None": pd.NA, "NULL": pd.NA}
    )
    return df
task_statuses_norm = cleaning_districts(task_statuses)

def cleaning_task_statuses(df):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if obj_cols:
        df[obj_cols] = df[obj_cols].astype("string")
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
        df[obj_cols] = df[obj_cols].replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA, "None": pd.NA, "NULL": pd.NA}
    )
    return df
task_types_norm = cleaning_districts(task_types)

dfs = {
    "assignments":    assignments_norm,
    "department":     departments_norm,
    "districts":      districts_norm,   
    "engineers":      engineers_norm,
    "equipments":      equipment_norm,
    "task_status":  task_statuses_norm,
    "task_types":     task_types_norm,
    "tasks":          tasks_norm,
}
cleaning_time = time.time()
print(f"{time.time() - start_time} seconds used for cleaning and normalizing dtypes.")


# ========= 这里开始改用 polars 做 join =========

# pandas -> polars
tasks_pl        = pl.from_pandas(dfs["tasks"])
assignments_pl  = pl.from_pandas(dfs["assignments"])
departments_pl  = pl.from_pandas(dfs["department"])
districts_pl    = pl.from_pandas(dfs["districts"])
engineers_pl    = pl.from_pandas(dfs["engineers"])
equipment_pl    = pl.from_pandas(dfs["equipments"])
task_status_pl  = pl.from_pandas(dfs["task_status"])
task_types_pl   = pl.from_pandas(dfs["task_types"])

# ---- 1. tasks 维表 join ----
import gc

# 如果前面那些 *_norm 变量还在，而且之后不再用，可以一并删
to_del = [
    "dfs",
    "assignments_norm", "departments_norm", "districts_norm",
    "engineers_norm", "equipment_norm",
    "task_statuses_norm", "task_types_norm", "tasks_norm",
]

for name in to_del:
    if name in globals():
        del globals()[name]

gc.collect()

# 重命名和原来保持一致
tasks_pl = tasks_pl.rename({
    "STATUS":     "STATUSID",
    "DEPARTMENT": "DEPARTMENTID",
    "DISTRICT":   "DISTRICTID",
    "TASKTYPE":   "TASKTYPEID",
    "W6KEY":      "TASK_KEY",
})

# 把 id 列转成 float（和你原来的行为一致）
id_cols = ["STATUSID", "DEPARTMENTID", "DISTRICTID", "TASKTYPEID"]
present = [c for c in id_cols if c in tasks_pl.columns]
if present:
    tasks_pl = tasks_pl.with_columns([pl.col(c).cast(pl.Float64) for c in present])

# 各个维表的 W6KEY → float
if "W6KEY" in task_status_pl.columns:
    task_status_pl = task_status_pl.with_columns(pl.col("W6KEY").cast(pl.Float64))
if "W6KEY" in task_types_pl.columns:
    task_types_pl = task_types_pl.with_columns(pl.col("W6KEY").cast(pl.Float64))
if "W6KEY" in districts_pl.columns:
    districts_pl = districts_pl.with_columns(pl.col("W6KEY").cast(pl.Float64))
if "W6KEY" in departments_pl.columns:
    departments_pl = departments_pl.with_columns(pl.col("W6KEY").cast(pl.Float64))

# tasks ← task_status
task_status_pl2 = task_status_pl.rename({"NAME": "STATUS"})
tasks_pl = (
    tasks_pl.join(
        task_status_pl2,
        left_on="STATUSID",
        right_on="W6KEY",
        how="left",
        suffix="_STATUS",
    )
)

# tasks ← task_types
task_types_pl2 = task_types_pl.rename({"NAME": "TASKTYPE"})
tasks_pl = (
    tasks_pl.join(
        task_types_pl2,
        left_on="TASKTYPEID",
        right_on="W6KEY",
        how="left",
        suffix="_TASKTYPE",
    )
)

# tasks ← districts
districts_pl2 = districts_pl.rename({"NAME": "DISTRICT"})
tasks_pl = (
    tasks_pl.join(
        districts_pl2,
        left_on="DISTRICTID",
        right_on="W6KEY",
        how="left",
        suffix="_DISTRICT",
    )
)

# tasks ← department
departments_pl2 = departments_pl.rename({"NAME": "DEPARTMENT"})
tasks_pl = (
    tasks_pl.join(
        departments_pl2,
        left_on="DEPARTMENTID",
        right_on="W6KEY",
        how="left",
        suffix="_DEPARTMENT",
    )
)

# 删掉中间的 ID 列（跟你原来一样）
drop_cols_tasks = [c for c in ["DEPARTMENTID","DISTRICTID","TASKTYPEID","STATUSID"] if c in tasks_pl.columns]
if drop_cols_tasks:
    tasks_pl = tasks_pl.drop(drop_cols_tasks)

# ---- 2. engineers ↔ assignments，后接 districts / department ----

engineers_pl = engineers_pl.rename({
    "DEPARTMENT": "DEPARTMENTID",
    "DISTRICT":   "DISTRICTID",
    "NAME":      "CREWNAME",
})

id_cols_eng = ["STATUSID", "DEPARTMENTID", "DISTRICTID", "TASKTYPEID"]
present_eng = [c for c in id_cols_eng if c in engineers_pl.columns]
if present_eng:
    engineers_pl = engineers_pl.with_columns([pl.col(c).cast(pl.Float64) for c in present_eng])

# assignments ← engineers  (full join)
if ("ASSIGNEDENGINEERS" in assignments_pl.columns) and ("CREWNAME" in engineers_pl.columns):
    assignments_pl = assignments_pl.join(
        engineers_pl,
        left_on="ASSIGNEDENGINEERS",
        right_on="CREWNAME",
        how="full",
        suffix="_ENGINEER",
    )

# assignments ← districts
if ("DISTRICTID" in assignments_pl.columns) and ("W6KEY" in districts_pl2.columns):
    assignments_pl = (
        assignments_pl.join(
            districts_pl2,
            left_on="DISTRICTID",
            right_on="W6KEY",
            how="left",
            suffix="_DISTRICT",
        )
    )

# assignments ← department
if ("DEPARTMENTID" in assignments_pl.columns) and ("W6KEY" in departments_pl2.columns):
    assignments_pl = (
        assignments_pl.join(
            departments_pl2,
            left_on="DEPARTMENTID",
            right_on="W6KEY",
            how="left",
            suffix="_DEPARTMENT",
        )
    )

drop_cols_assign = [c for c in ["DEPARTMENTID","DISTRICTID"] if c in assignments_pl.columns]
if drop_cols_assign:
    assignments_pl = assignments_pl.drop(drop_cols_assign)

# ---- 3. tasks ↔ assignments 按 TASK_KEY / TASK 做 full join ----

if "TASK_KEY" in tasks_pl.columns:
    tasks_pl = tasks_pl.with_columns(pl.col("TASK_KEY").cast(pl.Int64))
if "TASK" in assignments_pl.columns:
    assignments_pl = assignments_pl.with_columns(pl.col("TASK").cast(pl.Int64))

result_pl = tasks_pl.join(
    assignments_pl,
    left_on="TASK_KEY",
    right_on="TASK",
    how="full",
    suffix="_ASSIGNMENTS",
)



# polars -> pandas
result = result_pl.to_pandas()

dfs = {"tasks": result}
print(f"{time.time() - cleaning_time} seconds used for joining tables with polars.")

if report:
    generate_tables_report(dfs=dfs, out_html_path="Merge_tables_overview.html")

if saved:
    OUT_DIR = "data/cleaned_result"
    os.makedirs(OUT_DIR, exist_ok=True)
    dtype_book = {}
    for name, df in dfs.items():
        path = os.path.join(OUT_DIR, f"{name}.parquet")
        df.to_parquet(path, engine="pyarrow", index=False)
        dtype_book[name] = {col: str(dt) for col, dt in df.dtypes.items()}

    with open(os.path.join(OUT_DIR, "_dtype_book.json"), "w") as f:
        json.dump(dtype_book, f, indent=2)

