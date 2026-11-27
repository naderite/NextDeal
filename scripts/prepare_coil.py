"""Prepare UCI CoIL training portfolios for an isolated NextDeal runtime."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "https://kdd.ics.uci.edu/databases/tic/tic.html"
PRODUCTS = [
    "Private liability", "Business liability", "Agricultural liability", "Car", "Delivery van",
    "Motorcycle or scooter", "Lorry", "Trailer", "Tractor", "Agricultural machinery", "Moped",
    "Life", "Personal accident", "Family accident", "Disability", "Fire", "Surfboard", "Boat",
    "Bicycle", "Property", "Social security", "Mobile home",
]
FAMILIES = [
    "Liability", "Liability", "Liability", "Motor", "Motor", "Motor", "Motor", "Motor",
    "Motor", "Motor", "Motor", "Life and health", "Life and health", "Life and health",
    "Life and health", "Property", "Recreation", "Recreation", "Recreation", "Property",
    "Life and health", "Property",
]


def prepare(training, output):
    matrix = np.loadtxt(training, dtype=int, ndmin=2)
    if matrix.shape != (5822, 86):
        raise ValueError("Expected the publisher's original CoIL training table (5,822 by 86)")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    processed = output / "processed"
    pages = raw / "source_pages"
    pages.mkdir(parents=True)
    processed.mkdir()
    branches = dict(zip(PRODUCTS, FAMILIES, strict=True))
    profiles, companies, episodes = [], [], []
    for ref, row in enumerate(matrix, start=1):
        held = [p for p, count in zip(PRODUCTS, row[64:], strict=True) if count > 0]
        sector, category = f"Group {row[4]}", f"Subtype {row[0]}"
        profiles.append({"REF_pm": ref, "LIB_SECTEUR_ACTIVITE": sector, "LIB_ACTIVITE": category,
                         "category": category})
        companies.append({"REF_pm": ref, "sector": sector, "activity": category, "category": category,
                          "display_label": f"CoIL portfolio {ref}",
                          "provenance": "UCI CoIL 2000 training portfolio",
                          "portfolio": ";".join(held), "n_products": len(held),
                          "branches": ";".join(sorted({branches[p] for p in held})),
                          "has_history": 0})
        if len(held) > 1:
            for hidden in held:
                episodes.append({"REF_pm": ref, "hidden_product": hidden, "n_visible": len(held) - 1,
                                 "visible_portfolio": ";".join(p for p in held if p != hidden)})
    pd.DataFrame(profiles).to_csv(raw / "pm_product_portfolio.csv", sep=";", index=False)
    pd.DataFrame(companies).to_csv(processed / "pm_companies_clean.csv", index=False)
    pd.DataFrame(episodes).to_csv(processed / "pm_episodes.csv", index=False)
    (processed / "catalogue.json").write_text(
        json.dumps({"products": PRODUCTS, "product_branch": branches, "segmentation_category_weight": .5}, indent=2) + "\n"
    )
    # CoIL has no contract dates or payment outcomes.
    (raw / "pp_contract_renewals.json").write_text("[]\n")
    (processed / "pp_renewals_clean.csv").write_text("sub_branch,product,guarantee_list\n")
    (pages / "manifest.json").write_text(json.dumps([{"file": "coil.html", "url": SOURCE}]) + "\n")
    (pages / "coil.html").write_text(
        "<html><title>UCI CoIL 2000 benchmark</title><main>"
        "<p>CoIL 2000 is a consumer insurance portfolio research dataset. "
        "It records anonymized product holdings and area-level socioeconomic indicators. "
        "Product categories are research labels. Policy eligibility, terms, and coverage are unavailable."
        "</p></main></html>\n"
    )
    return len(companies), len(episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path, help="The publisher's ticdata2000.txt file")
    parser.add_argument("output", type=Path, help="A new dataset directory")
    args = parser.parse_args()
    clients, episodes = prepare(args.training, args.output)
    print(f"Prepared {clients} training portfolios and {episodes} ranking episodes in {args.output}")
