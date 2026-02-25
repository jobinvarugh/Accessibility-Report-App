# app.py (Streamlit) — Monthly Master Builder
from pathlib import Path
import tempfile
import os

import streamlit as st

from monthly_master_report import build_monthly_master_report


# ---------------------- SCHOOL → DEPT IDs ----------------------
SCHOOL_DEPT_IDS = {
    "Agrilife": ["568","569","570","571","572","573","574","575","576","577","578","579","580","581","582","583","805"],
    "Architecture": ["562","563","564","565","807"],
    "Arts Sciences": ["586","587","588","589","590","591","592","593","594","595","596","597","598","599","600","601","602","603","604","806"],
    "Bush School": ["644","645","646","647"],
    "CEHD": ["625","626","627","628","815"],
    "Dentistry": ["613","614","615","616","617","618","619","620","621","622","813"],
    "Engineering": ["551","629","630","631","632","633","634","635","636","637","638","639","640","641","642"],
    "Law": ["691"],
    "Marine Sciences": ["229","230","237","240","241","242","244","248","249","402","417","524","527"],
    "Mays School of Business": ["605","606","607","608","609","610"],
    "Medicine": ["546","550"],
    "Military Science": ["670","671","672","673"],
    "Nursing": ["675","680"],
    "Pharmacy": ["688","689"],
    "Public Health": ["681","682","683","684","685","686","814","818"],
    "PVFA": ["693"],
    "Qatar": ["343","344","345","346","702"],
    "Vet Med": ["695","696","697","698","699","700"],
}


def save_uploaded_file(uploaded_file) -> Path:
    """Save Streamlit UploadedFile to a temp file and return Path."""
    suffix = Path(uploaded_file.name).suffix or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


st.set_page_config(page_title="Monthly Accessibility Master Report", layout="wide")
st.title("Monthly Accessibility Master Report (Yearly Reset)")

st.markdown("### Upload files")

col1, col2 = st.columns(2)

with col1:
    prev_master_file = st.file_uploader(
        "Previous month master (.xlsx) — optional for first month",
        type=["xlsx"],
        key="prev_master",
    )
    ally_file = st.file_uploader("Current month Ally (csv/xlsx)", type=["csv", "xlsx"], key="ally")

with col2:
    pan_file = st.file_uploader("Current month Panorama (.xlsx)", type=["xlsx"], key="pan")

st.markdown("---")
st.markdown("### Settings")

report_month = st.text_input("Report month (YYYY-MM)", value="2026-02")

keep_only_prev = st.checkbox(
    "Keep only courses from previous master (recommended for your workflow)",
    value=True,
)

reset_year = st.text_input(
    "Reset year (optional) — drops month columns outside this year (e.g., 2026)",
    value="",
)

# ✅ College dropdown
school_options = ["All Colleges"] + list(SCHOOL_DEPT_IDS.keys())
selected_school = st.selectbox("College filter (optional)", school_options, index=0)

dept_ids_filter = None
if selected_school != "All Colleges":
    dept_ids_filter = SCHOOL_DEPT_IDS[selected_school]

term_filter = st.text_input("Term filter (optional)", value="")
pan_sheet = st.text_input("Panorama sheet override (optional)", value="")

# ✅ Zero enrollment dropdown
zero_opts = ["No", "Yes"]
exclude_zero_choice = st.selectbox("Exclude 0 enrollment courses?", zero_opts, index=0)
exclude_zero_enrollment = (exclude_zero_choice == "Yes")

st.markdown("---")
generate = st.button("Generate Master Excel", type="primary")

if generate:
    if not (ally_file and pan_file):
        st.error("Please upload the current month Ally and Panorama files.")
        st.stop()

    tmp_paths = []
    prev_master_path = None
    out_path = None

    try:
        if prev_master_file is not None:
            prev_master_path = save_uploaded_file(prev_master_file)
            tmp_paths.append(prev_master_path)

        ally_path = save_uploaded_file(ally_file)
        pan_path = save_uploaded_file(pan_file)
        tmp_paths.extend([ally_path, pan_path])

        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        out_path = Path(out_tmp.name)
        out_tmp.close()

        with st.spinner("Building monthly master report... (large files can take a bit)"):
            build_monthly_master_report(
                prev_master=prev_master_path,
                ally_current=ally_path,
                pan_current=pan_path,
                output_path=out_path,
                report_month=report_month.strip(),
                term_filter=term_filter.strip() or None,
                dept_ids_filter=dept_ids_filter,
                pan_sheet_override=pan_sheet.strip() or None,
                keep_only_prev_courses=keep_only_prev and (prev_master_path is not None),
                reset_to_year=reset_year.strip() or None,
                exclude_zero_enrollment=exclude_zero_enrollment,
            )

        with open(out_path, "rb") as f:
            st.success("Master report generated successfully.")
            fname = f"accessibility_master_{report_month.strip()}.xlsx"
            st.download_button(
                label="Download Master Excel",
                data=f,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.exception(e)

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        if out_path is not None:
            try:
                os.unlink(out_path)
            except Exception:
                pass
