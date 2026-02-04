# =========================================================
# GDP AUTOMATION PIPELINE (PRODUCTION)
# =========================================================

import requests
import pandas as pd
import itertools
from urllib.parse import quote
import logging
from datetime import datetime
import os
import sys
from google.cloud import bigquery
from google.oauth2 import service_account
import json

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
log_file = os.path.join(LOG_DIR, "pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

TIMEOUT = 30

# ---------------------------------------------------------
# BIGQUERY AUTH
# ---------------------------------------------------------
if "DATA_SERVICE_ACCOUNT_KEY" not in os.environ:
    raise EnvironmentError("❌ DATA_SERVICE_ACCOUNT_KEY secret олдсонгүй")

credentials_info = json.loads(os.environ["DATA_SERVICE_ACCOUNT_KEY"])

credentials = service_account.Credentials.from_service_account_info(
    credentials_info
)

bq_client = bigquery.Client(credentials=credentials)

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def get_table_metadata(table_path):
    encoded_path = quote(table_path, safe="/")
    url = f"https://data.1212.mn/api/v1/mn/NSO/{encoded_path}"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_nso_data(table_path, payload):
    encoded_path = quote(table_path, safe="/")
    url = f"https://data.1212.mn/api/v1/mn/NSO/{encoded_path}"
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def jsonstat_to_dataframe(data):
    dimensions = data["dimension"]
    values = data["value"]
    dim_names = data["id"]

    dim_labels = {}
    dim_sizes = []

    for dim in dim_names:
        labels = dimensions[dim]["category"]["label"]
        dim_labels[dim] = list(labels.values())
        dim_sizes.append(len(labels))

    rows = []
    for idx, combo in enumerate(itertools.product(*[range(s) for s in dim_sizes])):
        row = {}
        for i, dim in enumerate(dim_names):
            row[dim] = dim_labels[dim][combo[i]]
        row["DTVAL_CO"] = values[idx]
        rows.append(row)

    return pd.DataFrame(rows)


def pivot_validate(df, mapping, label):
    if "Бүрэлдэхүүн" not in df.columns:
        raise KeyError(f"{label}: 'Бүрэлдэхүүн' багана олдсонгүй")

    df["component"] = df["Бүрэлдэхүүн"].replace(mapping)

    pv = (
        df.pivot_table(
            index="ОН",
            columns="component",
            values="DTVAL_CO",
            aggfunc="first"
        )
        .reset_index()
    )

    ordered_cols = ["ОН"] + list(mapping.values())
    pv = pv.reindex(columns=ordered_cols)
    pv = pv.fillna(0)

    if pv.empty:
        raise ValueError(f"{label} pivot хоосон байна")

    missing = set(ordered_cols) - set(pv.columns)
    if missing:
        raise ValueError(f"{label} pivot багана дутуу: {missing}")

    logging.info(f"📊 {label} pivot OK")
    return pv

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    logging.info("🚀 GDP pipeline эхэллээ")

    table_path = "Economy, environment/National Accounts/DT_NSO_0500_022V1.px"
    metadata = get_table_metadata(table_path)

    def build_query(stat_code):
        query = {"query": [], "response": {"format": "json-stat2"}}
        for var in metadata["variables"]:
            if var["text"] == "Статистик үзүүлэлт":
                query["query"].append({
                    "code": var["code"],
                    "selection": {"filter": "item", "values": [stat_code]}
                })
            else:
                query["query"].append({
                    "code": var["code"],
                    "selection": {"filter": "item", "values": var["values"]}
                })
        return query

    # === MAPS ===
    ngdp_map = {
        "ДНБ": "ngdp",
        "Хөдөө аж ахуй, ойн аж ахуй, загас барилт, ан агнуур": "ngdp_agri",
        "Уул уурхай, олборлолт": "ngdp_mine",
        "Боловсруулах үйлдвэрлэл": "ngdp_manu",
        "Цахилгаан, хий, уур, агааржуулалт": "ngdp_elec",
        "Барилга": "ngdp_cons",
        "Бөөний болон жижиглэн худалдаа, машин, мотоциклийн засвар, үйлчилгээ": "ngdp_trad",
        "Тээвэр ба агуулахын үйл ажиллагаа": "ngdp_tran",
        "Мэдээлэл, холбоо": "ngdp_info",
        "Үйлчилгээний бусад үйл ажиллагаа": "ngdp_oser",
        "Бүтээгдэхүүний цэвэр татвар": "ngdp_taxe"
    }
    
    df_ngdp = jsonstat_to_dataframe(get_nso_data(table_path, build_query("0")))
    pv_ngdp = pivot_validate(df_ngdp, ngdp_map, "NGDP")
    # ===================== RGDP by 2005 =====================
    rgdp_2005_map = {k: f"rgdp_2005{v[4:]}" for k, v in ngdp_map.items()}
    
    df_rgdp_2005 = jsonstat_to_dataframe(get_nso_data(table_path, build_query("1")))
    pv_rgdp_2005 = pivot_validate(df_rgdp_2005, rgdp_2005_map, "RGDP 2005")
    
    # ===================== RGDP by 2010 =====================
    rgdp_2010_map = {k: f"rgdp_2010{v[4:]}" for k, v in ngdp_map.items()}
    
    df_rgdp_2010 = jsonstat_to_dataframe(get_nso_data(table_path, build_query("2")))
    pv_rgdp_2010 = pivot_validate(df_rgdp_2010, rgdp_2010_map, "RGDP 2010")
    
    # ===================== RGDP by 2015 =====================
    rgdp_2015_map = {k: f"rgdp_2015{v[4:]}" for k, v in ngdp_map.items()}
    
    df_rgdp_2015 = jsonstat_to_dataframe(get_nso_data(table_path, build_query("3")))
    pv_rgdp_2015 = pivot_validate(df_rgdp_2015, rgdp_2015_map, "RGDP 2015")

    # ===================== GROWTH =====================
    growth_map = {k: f"growth{v[4:]}" for k, v in ngdp_map.items()}
    
    df_growth = jsonstat_to_dataframe(get_nso_data(table_path, build_query("6")))
    pv_growth = pivot_validate(df_growth, growth_map, "GDP Growth")
        # ===================== POPULATION =====================
    pop_table_path = "Population, household/1_Population, household/DT_NSO_0300_003V1.px"

    pop_payload = {
        "query": [
            {"code": "Хүйс", "selection": {"filter": "item", "values": ["0", "1", "2"]}},
            {"code": "Насны бүлэг", "selection": {"filter": "item", "values": [str(i) for i in range(16)]}},
            {"code": "Он", "selection": {"filter": "item", "values": [str(i) for i in range(40)]}},
        ],
        "response": {"format": "json-stat2"}
    }

    df_population = jsonstat_to_dataframe(
        get_nso_data(pop_table_path, pop_payload)
    )

    pv_population = (
        df_population
        .pivot_table(
            index=["Хүйс", "Насны бүлэг"],
            columns="Он",
            values="DTVAL_CO",
            aggfunc="sum"
        )
        .reset_index()
    )
    pop_long = pv_population.melt(
    id_vars=["Хүйс", "Насны бүлэг"],
    var_name="year",
    value_name="value"
    )
    
    pop_long["indicator_code"] = "population"
    pop_long["source"] = "NSO 1212.mn"
    pop_long["loaded_at"] = pd.Timestamp.utcnow()
    pop_long["topic"] = "population"

    

    logging.info("📊 Population pivot OK")

    final_df = (
        pv_ngdp
        .merge(pv_rgdp_2005, on="ОН", how="outer")
        .merge(pv_rgdp_2010, on="ОН", how="outer")
        .merge(pv_rgdp_2015, on="ОН", how="outer")
        .merge(pv_growth, on="ОН", how="outer")
        )
    final_df = final_df.fillna(0)


    # ===================== EXPORT =====================
    if final_df.empty:
        raise ValueError("❌ final_df хоосон байна, экспорт хийх боломжгүй")
    
    # Багана дараалал (ОН эхэнд)
    cols = ["ОН"] + [c for c in final_df.columns if c != "ОН"]
    final_df = final_df[cols]
    
    output_file = os.path.join(
        OUTPUT_DIR,
        f"GDP_pipeline_{datetime.now().strftime('%Y%m%d')}.xlsx"
    )
    
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, sheet_name="GDP", index=False)
        pv_population.to_excel(writer, sheet_name="Population", index=False)

        # ===================== LOAD TO BIGQUERY (RAW, NO CHANGE) =====================
    table_id = "astute-azimuth-485909-p6.Automation_data.test_table"

    # Wide → Long (ямар ч drop / filter хийхгүй)
    id_col = "ОН"
    value_cols = [c for c in final_df.columns if c != id_col]

    long_df = final_df.melt(
        id_vars=id_col,
        value_vars=value_cols,
        var_name="indicator_code",
        value_name="value"
    )

    long_df = long_df.rename(columns={"ОН": "year"})
    long_df["source"] = "NSO 1212.mn"
    long_df["loaded_at"] = pd.Timestamp.utcnow()
    long_df["topic"] = "gdp" #Sheet name option
    # ===================== POPULATION → LONG =====================
    pop_long = pv_population.melt(
        id_vars=["Хүйс", "Насны бүлэг"],
        var_name="year",
        value_name="value"
    )
    
    pop_long = pop_long.rename(columns={
        "Хүйс": "sex",
        "Насны бүлэг": "age_group"
    })
    
    pop_long["topic"] = "population"
    pop_long["source"] = "NSO 1212.mn"
    pop_long["loaded_at"] = pd.Timestamp.utcnow()

    pop_long["topic"] = "population"

    # ===================== FINAL MERGE =====================
    final_long = pd.concat(
        [long_df, pop_long],
        ignore_index=True
    )
    
    job = bq_client.load_table_from_dataframe(
        final_long,
        table_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE"
        )
    )



    job.result()
    logging.info(f"☁️ BigQuery-д {len(final_long)} мөр (GDP + Population) бичигдлээ")

    
    logging.info(f"✅ Pipeline амжилттай дууслаа → {output_file}")

# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
