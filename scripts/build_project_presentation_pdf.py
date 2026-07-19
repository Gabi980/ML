from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
TMP = ROOT / "tmp" / "pdfs"
OUT = ROOT / "output" / "pdf"
PDF_PATH = OUT / "prezentare_proiect_machine_learning.pdf"


def register_fonts() -> dict[str, str]:
    fonts = {
        "regular": "Helvetica",
        "bold": "Helvetica-Bold",
        "mono": "Courier",
    }
    font_files = {
        "regular": Path(r"C:\Windows\Fonts\arial.ttf"),
        "bold": Path(r"C:\Windows\Fonts\arialbd.ttf"),
        "mono": Path(r"C:\Windows\Fonts\consola.ttf"),
    }
    if font_files["regular"].exists():
        pdfmetrics.registerFont(TTFont("Arial", str(font_files["regular"])))
        fonts["regular"] = "Arial"
    if font_files["bold"].exists():
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_files["bold"])))
        fonts["bold"] = "Arial-Bold"
    if font_files["mono"].exists():
        pdfmetrics.registerFont(TTFont("Consolas", str(font_files["mono"])))
        fonts["mono"] = "Consolas"
    return fonts


FONTS = register_fonts()


def make_styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=base["Title"],
            fontName=FONTS["bold"],
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324d"),
            spaceAfter=18,
        )
    )
    base.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=base["Normal"],
            fontName=FONTS["regular"],
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#405466"),
            spaceAfter=12,
        )
    )
    base.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=base["Heading1"],
            fontName=FONTS["bold"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=8,
            spaceAfter=10,
        )
    )
    base.add(
        ParagraphStyle(
            name="Subsection",
            parent=base["Heading2"],
            fontName=FONTS["bold"],
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#235789"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="Body",
            parent=base["BodyText"],
            fontName=FONTS["regular"],
            fontSize=9.7,
            leading=13.2,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="Small",
            parent=base["BodyText"],
            fontName=FONTS["regular"],
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#4d5b66"),
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="Caption",
            parent=base["BodyText"],
            fontName=FONTS["regular"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#5c6770"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=7,
        )
    )
    base.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=base["Code"],
            fontName=FONTS["mono"],
            fontSize=7.2,
            leading=9.2,
            textColor=colors.HexColor("#1f2a33"),
            backColor=colors.HexColor("#f3f6f8"),
            borderPadding=6,
            leftIndent=0,
            rightIndent=0,
        )
    )
    base.add(
        ParagraphStyle(
            name="Callout",
            parent=base["BodyText"],
            fontName=FONTS["regular"],
            fontSize=9.5,
            leading=12.5,
            textColor=colors.HexColor("#17324d"),
            backColor=colors.HexColor("#edf5fb"),
            borderColor=colors.HexColor("#9cc8e5"),
            borderWidth=0.6,
            borderPadding=7,
            spaceBefore=6,
            spaceAfter=8,
        )
    )
    return base


STYLES = make_styles()


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item), bulletColor=colors.HexColor("#235789")) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
        bulletFontName=FONTS["regular"],
        bulletFontSize=8,
    )


def df_table(df: pd.DataFrame, columns: list[str], labels: list[str], widths: list[float]) -> Table:
    table_data = [labels]
    for _, row in df[columns].iterrows():
        out = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if abs(value) >= 1000:
                    out.append(f"{value:,.0f}")
                else:
                    out.append(f"{value:.4f}")
            else:
                out.append(str(value))
        table_data.append(out)
    table = Table(table_data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONTS["bold"]),
                ("FONTNAME", (0, 1), (-1, -1), FONTS["regular"]),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("LEADING", (0, 0), (-1, -1), 9.2),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7d0d8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def image_flowable(path: Path, max_width: float = 16.0 * cm, max_height: float = 9.0 * cm) -> Image:
    reader = ImageReader(str(path))
    width_px, height_px = reader.getSize()
    ratio = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio)


def add_figure(story: list, path: Path, caption: str, max_width: float = 16.0 * cm, max_height: float = 9.0 * cm) -> None:
    story.append(KeepTogether([image_flowable(path, max_width, max_height), p(caption, "Caption")]))


def make_extra_figures() -> dict[str, Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    house = pd.read_csv(TABLES / "house_price_regression_model_comparison.csv")
    telco = pd.read_csv(TABLES / "telco_churn_classification_model_comparison.csv")
    cm = pd.read_csv(TABLES / "telco_churn_confusion_matrix.csv", index_col=0)

    paths = {
        "regression_bar": TMP / "regression_rmse_comparison.png",
        "classification_bar": TMP / "classification_f1_comparison.png",
        "classification_auc": TMP / "classification_auc_comparison.png",
        "confusion_matrix": TMP / "svm_confusion_matrix.png",
        "pipeline": TMP / "project_pipeline.png",
    }

    plt.figure(figsize=(10, 5))
    h = house.sort_values("holdout_rmse", ascending=False)
    colors_bar = ["#4f9fd6" if group == "ensemble" else "#9bb7c9" for group in h["model_group"]]
    plt.barh(h["model_name"], h["holdout_rmse"], color=colors_bar)
    plt.title("Regresie - comparatie modele dupa RMSE holdout")
    plt.xlabel("RMSE (mai mic este mai bun)")
    plt.tight_layout()
    plt.savefig(paths["regression_bar"], dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    t = telco.sort_values("holdout_f1")
    colors_bar = ["#4f9fd6" if group == "ensemble" else "#9bb7c9" for group in t["model_group"]]
    plt.barh(t["model_name"], t["holdout_f1"], color=colors_bar)
    plt.title("Clasificare - comparatie modele dupa F1 holdout")
    plt.xlabel("F1 score (mai mare este mai bun)")
    plt.xlim(0, max(0.75, t["holdout_f1"].max() + 0.05))
    plt.tight_layout()
    plt.savefig(paths["classification_bar"], dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    t = telco.sort_values("holdout_roc_auc")
    colors_bar = ["#4f9fd6" if group == "ensemble" else "#9bb7c9" for group in t["model_group"]]
    plt.barh(t["model_name"], t["holdout_roc_auc"], color=colors_bar)
    plt.title("Clasificare - comparatie modele dupa ROC AUC holdout")
    plt.xlabel("ROC AUC (mai mare este mai bun)")
    plt.xlim(0.55, 0.9)
    plt.tight_layout()
    plt.savefig(paths["classification_auc"], dpi=180)
    plt.close()

    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matrice de confuzie - SVM")
    plt.ylabel("Valoare reala")
    plt.xlabel("Predictie")
    plt.tight_layout()
    plt.savefig(paths["confusion_matrix"], dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    labels = [
        "CSV Kaggle",
        "Curatare date",
        "EDA si grafice",
        "Preprocesare",
        "Training modele",
        "Comparare metrici",
        "Salvare model",
        "GUI Streamlit",
    ]
    x_positions = [i / (len(labels) - 1) for i in range(len(labels))]
    for i, (x, label) in enumerate(zip(x_positions, labels)):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#17324d",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#edf5fb", edgecolor="#4f9fd6"),
        )
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.055, 0.5),
                xytext=(x + 0.055, 0.5),
                arrowprops=dict(arrowstyle="->", color="#4f9fd6", lw=1.5),
            )
    plt.tight_layout()
    plt.savefig(paths["pipeline"], dpi=180)
    plt.close()

    return paths


def read_code_snippet(path: Path, start_marker: str, end_marker: str | None = None, max_lines: int = 42) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if start_marker in line:
            start = i
            break
    end = min(len(lines), start + max_lines)
    if end_marker:
        for i in range(start + 1, len(lines)):
            if end_marker in lines[i]:
                end = i
                break
    snippet = "\n".join(lines[start:end]).rstrip()
    return snippet


def code_block(code: str) -> Preformatted:
    return Preformatted(code, STYLES["CodeBlock"], maxLineLength=92)


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFont(FONTS["regular"], 8)
    canvas.setFillColor(colors.HexColor("#687887"))
    canvas.drawString(1.5 * cm, 1.0 * cm, "Proiect Machine Learning - prezentare tehnica")
    canvas.drawRightString(width - 1.5 * cm, 1.0 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    extra = make_extra_figures()
    house = pd.read_csv(TABLES / "house_price_regression_model_comparison.csv")
    telco = pd.read_csv(TABLES / "telco_churn_classification_model_comparison.csv")
    house_summary = pd.read_csv(TABLES / "house_sales_in_king_county_statistical_summary.csv", index_col=0)
    telco_summary = pd.read_csv(TABLES / "telco_customer_churn_statistical_summary.csv", index_col=0)
    house_corr = pd.read_csv(TABLES / "house_sales_in_king_county_top_target_correlations.csv", header=None, names=["feature", "correlation"])

    story: list = []

    story.append(Spacer(1, 2.3 * cm))
    story.append(p("Platforma de predicții de business folosind Machine Learning", "CoverTitle"))
    story.append(p("Estimarea prețului locuințelor și predicția pierderii clienților", "CoverSubtitle"))
    story.append(Spacer(1, 0.6 * cm))
    story.append(p("Proiect Machine Learning - 2025-2026", "CoverSubtitle"))
    story.append(Spacer(1, 1.2 * cm))
    story.append(
        p(
            "Aplicația demonstrează un pipeline complet de Machine Learning pentru două tipuri de probleme cerute în proiect: regresie și clasificare. "
            "Modulul de regresie estimează prețul unei case din King County, iar modulul de clasificare estimează riscul ca un client telecom să renunțe la serviciu.",
            "Callout",
        )
    )
    add_figure(story, extra["pipeline"], "Figura 1. Fluxul general al proiectului, de la dataseturile Kaggle până la GUI-ul Streamlit.", 16 * cm, 4 * cm)
    story.append(PageBreak())

    story.append(p("1. Scopul aplicației", "SectionTitle"))
    story.append(
        p(
            "Scopul proiectului este să arate cum pot fi folosite metodele studiate la materia Machine Learning într-un produs minimal, dar complet. "
            "Aplicația nu încearcă să lege direct cele două dataseturi, deoarece ele provin din domenii diferite. Legătura dintre ele este metodologică: același mod de lucru este aplicat unei probleme de regresie și unei probleme de clasificare.",
        )
    )
    story.append(
        bullets(
            [
                "Pentru regresie, intrarea este un set de caracteristici ale unei locuințe, iar ieșirea este un preț estimat.",
                "Pentru clasificare, intrarea este profilul unui client telecom, iar ieșirea este clasa prezisă: clientul pleacă sau nu pleacă.",
                "Pentru ambele module, codul face analiză statistică, preprocesare, antrenare, evaluare comparativă și salvare de model.",
                "GUI-ul Streamlit folosește modelele salvate și permite introducerea unor exemple noi prin formulare.",
            ]
        )
    )
    story.append(p("Cerințe acoperite", "Subsection"))
    story.append(
        bullets(
            [
                "Două dataseturi: unul pentru regresie și unul pentru clasificare.",
                "Sumare statistice și vizualizări pentru fiecare dataset.",
                "Transformări de preprocesare prezentate și aplicate în pipeline.",
                "Câte 5 modele de bază pentru fiecare problemă.",
                "Câte 2 metode ensemble pentru fiecare problemă.",
                "Salvarea celui mai bun model și deploy minimal prin GUI.",
            ]
        )
    )

    story.append(p("2. Structura implementării", "SectionTitle"))
    story.append(
        p(
            "Proiectul este separat pe responsabilități. Această structură face codul mai ușor de explicat și de modificat: partea de încărcare a datelor este separată de modelare, raportare și interfață.",
        )
    )
    structure_rows = [
        ["Fișier", "Rol"],
        ["src/data_preparation.py", "Încarcă dataseturile, curăță coloanele, convertește targetul și separă X/y."],
        ["src/modeling.py", "Definește modelele, preprocesarea, cross-validation, metricile și salvarea artefactelor."],
        ["src/reporting.py", "Generează sumare statistice, tabele CSV, grafice EDA și sumarul proiectului."],
        ["src/train.py", "Entry point: rulează pipeline-ul complet pentru ambele dataseturi."],
        ["app.py", "Interfața Streamlit cu două taburi pentru predicții noi."],
    ]
    table = Table(structure_rows, colWidths=[5.2 * cm, 10.4 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONTS["bold"]),
                ("FONTNAME", (0, 1), (-1, -1), FONTS["regular"]),
                ("FONTSIZE", (0, 0), (-1, -1), 8.1),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7d0d8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(PageBreak())

    story.append(p("3. Datasetul de regresie: House Sales in King County", "SectionTitle"))
    story.append(
        p(
            "Datasetul conține vânzări reale de case din King County, Washington. Targetul este coloana `price`, adică prețul de vânzare al locuinței. "
            "După curățare au rămas 21.611 observații și 22 de coloane, deoarece data vânzării a fost transformată în componente utile: an, lună și zi din săptămână.",
        )
    )
    mini_house = (
        house_summary.loc[[idx for idx in ["price", "bedrooms", "bathrooms", "sqft_living", "grade", "yr_built"] if idx in house_summary.index]]
        [["count", "mean", "std", "min", "50%", "max"]]
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    story.append(p("Sumar statistic relevant", "Subsection"))
    story.append(
        df_table(
            mini_house,
            ["feature", "count", "mean", "std", "min", "50%", "max"],
            ["Feature", "Count", "Mean", "Std", "Min", "Median", "Max"],
            [3.2 * cm, 2.1 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm, 2.1 * cm, 2.2 * cm],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    add_figure(story, FIGURES / "house_sales_in_king_county_target_distribution.png", "Figura 2. Distribuția targetului `price`. Se observă valori extreme, lucru normal pentru prețuri imobiliare.", 15.5 * cm, 7.2 * cm)
    add_figure(story, FIGURES / "house_sales_in_king_county_sqft_living_vs_price.png", "Figura 3. Relația dintre suprafața locuibilă și preț. Relația este pozitivă, dar nu perfect liniară.", 15.5 * cm, 7.2 * cm)
    story.append(PageBreak())

    story.append(p("Corelații și interpretare pentru regresie", "Subsection"))
    story.append(
        p(
            "Corelația este utilă pentru a observa rapid ce variabile au legătură liniară cu prețul. Cele mai importante corelații apar de obicei pentru suprafață, gradul construcției, calitatea zonei și numărul de băi.",
        )
    )
    story.append(
        df_table(
            house_corr.head(10),
            ["feature", "correlation"],
            ["Feature", "Corelație cu price"],
            [8.0 * cm, 4.0 * cm],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    add_figure(story, FIGURES / "house_sales_in_king_county_top_target_correlations.png", "Figura 4. Cele mai mari corelații absolute cu prețul casei.", 15.5 * cm, 7.0 * cm)
    add_figure(story, FIGURES / "house_sales_in_king_county_correlation_matrix.png", "Figura 5. Matricea de corelații pentru variabilele numerice din datasetul de case.", 14.0 * cm, 10.5 * cm)
    story.append(PageBreak())

    story.append(p("4. Datasetul de clasificare: Telco Customer Churn", "SectionTitle"))
    story.append(
        p(
            "Datasetul Telco descrie clienți ai unei companii de telecomunicații. Targetul este `Churn`, convertit în 0 pentru `No` și 1 pentru `Yes`. "
            "Problema este de clasificare binară și este ușor dezechilibrată: clienții care nu pleacă sunt mai numeroși decât cei care pleacă.",
        )
    )
    mini_telco = (
        telco_summary.loc[[idx for idx in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"] if idx in telco_summary.index]]
        [["count", "mean", "std", "min", "50%", "max"]]
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    story.append(p("Sumar statistic relevant", "Subsection"))
    story.append(
        df_table(
            mini_telco,
            ["feature", "count", "mean", "std", "min", "50%", "max"],
            ["Feature", "Count", "Mean", "Std", "Min", "Median", "Max"],
            [3.2 * cm, 2.1 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm, 2.1 * cm, 2.2 * cm],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    add_figure(story, FIGURES / "telco_customer_churn_target_distribution.png", "Figura 6. Distribuția claselor pentru `Churn`.", 13.5 * cm, 6.2 * cm)
    add_figure(story, FIGURES / "telco_customer_churn_contract_churn_rate.png", "Figura 7. Rata churn în funcție de tipul contractului.", 14.5 * cm, 6.3 * cm)
    story.append(PageBreak())

    story.append(p("Vizualizări suplimentare pentru churn", "Subsection"))
    story.append(
        p(
            "Vizualizările arată că variabilele contractuale și comportamentul de facturare sunt foarte relevante. Clienții cu contracte mai flexibile sau metode de plată mai puțin stabile tind să aibă risc mai mare de churn.",
        )
    )
    add_figure(story, FIGURES / "telco_customer_churn_paymentmethod_churn_rate.png", "Figura 8. Rata churn în funcție de metoda de plată.", 15.0 * cm, 6.5 * cm)
    add_figure(story, FIGURES / "telco_customer_churn_tenure_by_churn.png", "Figura 9. Distribuția vechimii clientului în funcție de churn.", 15.0 * cm, 6.5 * cm)
    add_figure(story, FIGURES / "telco_customer_churn_monthlycharges_by_churn.png", "Figura 10. Distribuția costului lunar în funcție de churn.", 15.0 * cm, 6.5 * cm)
    story.append(PageBreak())

    story.append(p("5. Preprocesarea datelor", "SectionTitle"))
    story.append(
        p(
            "Preprocesarea este implementată ca parte din pipeline-ul scikit-learn. Acest lucru este important deoarece aceleași transformări aplicate pe datele de training sunt aplicate automat și pe datele noi introduse în GUI.",
        )
    )
    story.append(
        bullets(
            [
                "Identificatorii fără valoare predictivă (`id`, `customerID`) sunt eliminați.",
                "Coloana `date` din datasetul de case este transformată în `sale_year`, `sale_month` și `sale_dayofweek`.",
                "`TotalCharges` este convertită din text în numeric, iar eticheta `Churn` este mapată la 0/1.",
                "Valorile lipsă numerice sunt imputate cu mediana; valorile categorice cu moda.",
                "Variabilele numerice sunt standardizate cu `StandardScaler`.",
                "Variabilele categorice sunt transformate prin `OneHotEncoder(handle_unknown='ignore')`.",
            ]
        )
    )
    story.append(p("Fragment de cod - preprocesare", "Subsection"))
    story.append(code_block(read_code_snippet(ROOT / "src" / "modeling.py", "def build_preprocessor", "def regression_models")))

    story.append(p("6. Modele antrenate", "SectionTitle"))
    story.append(
        p(
            "Pentru fiecare dataset au fost antrenate cinci modele de bază și două modele ensemble. Modelele au fost evaluate prin cross-validation pe setul de training și apoi pe un holdout test set.",
        )
    )
    story.append(p("Modele pentru regresie", "Subsection"))
    story.append(
        bullets(
            [
                "Linear Regression, Ridge Regression, Lasso Regression.",
                "KNN Regressor și Decision Tree Regressor.",
                "Ensemble: Random Forest Regressor și AdaBoost Regressor.",
            ]
        )
    )
    story.append(p("Modele pentru clasificare", "Subsection"))
    story.append(
        bullets(
            [
                "Logistic Regression, KNN Classifier, Gaussian Naive Bayes.",
                "Decision Tree Classifier și Support Vector Machine.",
                "Ensemble: Random Forest Classifier și AdaBoost Classifier.",
            ]
        )
    )
    story.append(p("Fragment de cod - lista modelelor", "Subsection"))
    story.append(code_block(read_code_snippet(ROOT / "src" / "modeling.py", "def regression_models", "def build_feature_schema", max_lines=80)))
    story.append(PageBreak())

    story.append(p("7. Comparația modelelor de regresie", "SectionTitle"))
    story.append(
        p(
            "Pentru regresie au fost urmărite MAE, RMSE și R2. RMSE penalizează mai puternic erorile mari, de aceea este o metrică bună pentru prețuri imobiliare, unde predicțiile foarte greșite sunt costisitoare.",
        )
    )
    story.append(
        df_table(
            house,
            ["model_group", "model_name", "holdout_mae", "holdout_rmse", "holdout_r2"],
            ["Tip", "Model", "MAE", "RMSE", "R2"],
            [2.0 * cm, 5.2 * cm, 3.0 * cm, 3.0 * cm, 2.2 * cm],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    add_figure(story, extra["regression_bar"], "Figura 11. Comparația modelelor de regresie după RMSE pe holdout set.", 15.8 * cm, 7.2 * cm)
    story.append(
        p(
            "Modelul selectat este Random Forest Regressor. Acesta obține cel mai mic RMSE și cel mai mare R2, ceea ce indică faptul că surprinde mai bine relațiile neliniare dintre caracteristicile caselor și preț.",
            "Callout",
        )
    )
    story.append(PageBreak())

    story.append(p("8. Comparația modelelor de clasificare", "SectionTitle"))
    story.append(
        p(
            "Pentru clasificare au fost urmărite accuracy, precision, recall, F1 și ROC AUC. Într-o problemă de churn, F1 este foarte util deoarece combină precision și recall. "
            "Accuracy singură poate fi înșelătoare atunci când clasele nu sunt perfect echilibrate.",
        )
    )
    story.append(
        df_table(
            telco,
            ["model_group", "model_name", "holdout_accuracy", "holdout_precision", "holdout_recall", "holdout_f1", "holdout_roc_auc"],
            ["Tip", "Model", "Acc", "Prec", "Recall", "F1", "AUC"],
            [1.7 * cm, 4.4 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm],
        )
    )
    add_figure(story, extra["classification_bar"], "Figura 12. Comparația modelelor de clasificare după F1 pe holdout set.", 15.8 * cm, 7.0 * cm)
    add_figure(story, extra["classification_auc"], "Figura 13. Comparația modelelor de clasificare după ROC AUC pe holdout set.", 15.8 * cm, 7.0 * cm)
    story.append(PageBreak())

    story.append(p("Interpretarea modelului de clasificare selectat", "Subsection"))
    story.append(
        p(
            "Modelul selectat după F1 este Support Vector Machine. Are recall ridicat pentru clasa `Yes`, ceea ce înseamnă că identifică o mare parte dintre clienții care chiar pleacă. "
            "În aplicații de retenție, acest comportament este util: este de preferat să detectăm cât mai mulți clienți cu risc, chiar dacă apar și câteva alarme false.",
        )
    )
    add_figure(story, extra["confusion_matrix"], "Figura 14. Matricea de confuzie pentru modelul SVM pe holdout set.", 11.0 * cm, 8.0 * cm)
    story.append(
        p(
            "Matricea arată 287 clienți churn detectați corect și 85 clienți churn ratați. Există 253 false pozitive, adică persoane marcate cu risc de churn deși nu au plecat. "
            "Acest compromis este acceptabil dacă scopul business este prevenirea pierderii clienților, nu doar maximizarea acurateții globale.",
            "Callout",
        )
    )
    story.append(PageBreak())

    story.append(p("9. Ensemble methods", "SectionTitle"))
    story.append(
        p(
            "Ensemble methods combină mai multe modele pentru a obține predicții mai stabile. În proiect au fost folosite Random Forest și AdaBoost pentru ambele tipuri de probleme.",
        )
    )
    story.append(
        bullets(
            [
                "Random Forest construiește mai mulți arbori pe subsample-uri ale datelor și agregă predicțiile. Reduce varianța specifică arborilor de decizie.",
                "AdaBoost construiește modele succesive care încearcă să corecteze erorile modelelor anterioare.",
                "La regresie, Random Forest a fost clar cel mai bun model.",
                "La clasificare, ensemble-urile au avut accuracy bună, dar SVM a obținut F1 mai bun datorită recall-ului mai ridicat.",
            ]
        )
    )
    ensemble_house = house[house["model_group"] == "ensemble"]
    ensemble_telco = telco[telco["model_group"] == "ensemble"]
    story.append(p("Rezultate ensemble - regresie", "Subsection"))
    story.append(
        df_table(
            ensemble_house,
            ["model_name", "holdout_mae", "holdout_rmse", "holdout_r2"],
            ["Model", "MAE", "RMSE", "R2"],
            [5.5 * cm, 3.2 * cm, 3.2 * cm, 2.2 * cm],
        )
    )
    story.append(p("Rezultate ensemble - clasificare", "Subsection"))
    story.append(
        df_table(
            ensemble_telco,
            ["model_name", "holdout_accuracy", "holdout_precision", "holdout_recall", "holdout_f1", "holdout_roc_auc"],
            ["Model", "Acc", "Prec", "Recall", "F1", "AUC"],
            [4.5 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm],
        )
    )
    story.append(PageBreak())

    story.append(p("10. Salvarea modelelor și GUI-ul Streamlit", "SectionTitle"))
    story.append(
        p(
            "După evaluare, pipeline-ul complet al celui mai bun model este salvat cu `joblib`. Artefactul salvat conține nu doar estimatorul, ci și preprocesarea și schema de input folosită de aplicație.",
        )
    )
    story.append(
        bullets(
            [
                "`outputs/models/house_price_best_model.joblib` - modelul Random Forest Regressor.",
                "`outputs/models/telco_churn_best_model.joblib` - modelul Support Vector Machine.",
                "`app.py` - aplicație Streamlit cu două taburi: House Price Regression și Telco Churn Classification.",
                "Formularele sunt generate din schema de feature-uri salvată în artefact, astfel încât interfața rămâne sincronizată cu modelul.",
            ]
        )
    )
    story.append(p("Fragment de cod - salvare artefact și GUI", "Subsection"))
    story.append(code_block(read_code_snippet(ROOT / "src" / "modeling.py", "def save_artifact", max_lines=8)))
    story.append(Spacer(1, 0.25 * cm))
    story.append(code_block(read_code_snippet(ROOT / "app.py", "def render_input_form", "def render_regression_tab", max_lines=52)))
    story.append(PageBreak())

    story.append(p("11. Limitări și direcții de îmbunătățire", "SectionTitle"))
    story.append(
        bullets(
            [
                "Pentru regresie, prețurile imobiliare au outlieri puternici; o variantă îmbunătățită ar putea modela `log(price)`.",
                "Pentru churn, alegerea metricii influențează modelul final. Dacă obiectivul ar fi maximizarea profitului, ar trebui introdusă o funcție de cost business.",
                "SVM a fost selectat după F1, dar pentru producție ar fi utilă calibrarea probabilităților.",
                "GUI-ul este minimal: bun pentru demonstrație, dar nu include autentificare, logging, monitorizare drift sau bază de date.",
                "Random Forest este performant, dar modelul salvat pentru case este mare; pentru deploy real ar putea fi testate modele mai compacte sau optimizare de hiperparametri.",
            ]
        )
    )
    story.append(p("12. Concluzie", "SectionTitle"))
    story.append(
        p(
            "Proiectul acoperă complet fluxul studiat la laborator: date, analiză, preprocesare, modele, evaluare, ensemble-uri, salvare și interfață. "
            "Cele două module nu sunt legate prin același domeniu de date, ci prin aceeași metodologie de Machine Learning aplicată la două tipuri diferite de task-uri. "
            "Rezultatul este o aplicație demonstrativă care poate prezice atât o valoare numerică, cât și o clasă, folosind modele antrenate și salvate local.",
            "Callout",
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(p("Comenzi utile", "Subsection"))
    story.append(
        code_block(
            ".\\.venv\\Scripts\\Activate.ps1\n"
            "python -m src.train\n"
            "streamlit run app.py"
        )
    )

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.55 * cm,
        title="Prezentare Proiect Machine Learning",
        author="Codex",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return PDF_PATH


if __name__ == "__main__":
    print(build_pdf())
