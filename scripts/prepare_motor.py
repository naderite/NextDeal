"""Prepare a historical renewal snapshot from the Lledó and Pavía motor portfolio."""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

SOURCE = "https://doi.org/10.17632/5cxyb5fp4f.2"
RISK = {1: "Motorbike", 2: "Van", 3: "Passenger car", 4: "Agricultural vehicle"}


def prepare(path, output, as_of, limit, minimum_lead_days):
    fields = ["ID", "Date_last_renewal", "Date_next_renewal", "Date_lapse", "Type_risk"]
    rows = pd.read_csv(path, sep=None, engine="python", usecols=fields, dtype={"ID": str})
    for field in ("Date_last_renewal", "Date_next_renewal", "Date_lapse"):
        rows[field] = pd.to_datetime(rows[field], format="%d/%m/%Y", errors="raise")
    day = pd.Timestamp(as_of)
    eligible = rows.loc[
        rows.Date_last_renewal.le(day)
        & rows.Date_next_renewal.gt(day)
        & rows.Date_next_renewal.ge(day + pd.Timedelta(days=minimum_lead_days))
        & (rows.Date_lapse.isna() | rows.Date_lapse.gt(day))
    ]
    snapshot = (eligible.sort_values(["Date_last_renewal", "ID"])
                .drop_duplicates("ID", keep="last").sort_values(["Date_next_renewal", "ID"]))
    if limit:
        snapshot = snapshot.head(limit)
    if snapshot.empty:
        raise ValueError("No observed active policyholder-product records for that date")
    output.mkdir(parents=True, exist_ok=False)
    (output / "dataset.json").write_text(json.dumps({"demo_date": as_of.isoformat(), "source": SOURCE}) + "\n")
    raw, processed = output / "raw", output / "processed"
    pages = raw / "source_pages"
    pages.mkdir(parents=True)
    processed.mkdir()
    contracts, products = [], []
    for row in snapshot.itertuples(index=False):
        product = RISK[int(row.Type_risk)]
        products.append({"sub_branch": "Motor", "product": product})
        contracts.append({"user_id": row.ID, "profile": {
            "Num_Contrat": row.ID, "LIB_PRODUIT": product, "LIB_BRANCHE": "Motor",
            "LIB_SOUS_BRANCHE": "Motor", "Effet_Contrat": row.Date_last_renewal.date().isoformat(),
            "Date_Expiration": row.Date_next_renewal.date().isoformat(),
            "statut_paiement": "Unknown", "Lib_Garantie": "",
            "provenance": "Lledó and Pavía motor portfolio; expiry field represents scheduled renewal"
        }})
    (raw / "pp_contract_renewals.json").write_text(json.dumps(contracts, ensure_ascii=False) + "\n")
    pd.DataFrame(products).to_csv(processed / "pp_renewals_clean.csv", index=False)
    (processed / "pm_companies_clean.csv").write_text("REF_pm,category,branches,n_products,has_history\n")
    (processed / "pm_episodes.csv").write_text("REF_pm,hidden_product,visible_portfolio,n_visible\n")
    (processed / "catalogue.json").write_text('{"products": [], "product_branch": {}}\n')
    (pages / "manifest.json").write_text(json.dumps([{"file": "portfolio.html", "url": SOURCE}]) + "\n")
    (pages / "portfolio.html").write_text(
        "<html><title>Spanish motor portfolio benchmark</title><main>"
        "<p>The motor insurance portfolio contains anonymized policyholder-product identifiers and "
        "annual renewal dates. The extract does not document policy wording or individual invoice outcomes."
        "</p></main></html>\n"
    )
    return len(contracts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", type=Path)
    parser.add_argument("output", type=Path, help="New dataset directory")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2017, 12, 31))
    parser.add_argument("--limit", type=int, default=500, help="Earliest renewals to load, or 0 for all")
    parser.add_argument("--minimum-lead-days", type=int, default=30, help="Minimum days until renewal on the as-of date")
    args = parser.parse_args()
    if args.limit < 0 or args.minimum_lead_days < 0:
        parser.error("--limit and --minimum-lead-days must be nonnegative")
    print(f"Prepared {prepare(args.portfolio, args.output, args.as_of, args.limit, args.minimum_lead_days)} motor renewals")
