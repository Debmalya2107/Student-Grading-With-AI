
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Optional
import xlsxwriter
import openpyxl


def main():
    st.set_page_config(
        page_title="Student Marks Analyzer",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
     body {
            background-color: #f4f6f9;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .title-box {
            background: linear-gradient(90deg, #4CAF50, #2E7D32);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        .upload-box {
            background: white;
            border: 2px dashed #4CAF50;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }
        .stDownloadButton button {
            background: #4CAF50 !important;
            color: white !important;
            border-radius: 10px !important;
            font-size: 18px !important;
            padding: 10px 20px !important;
        }
        .main > div {
            padding-top: 2rem;
        }
        .stAlert > div {
            padding-top: 1rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title(" Student Marks Analyzer")
    st.markdown("**Upload an Excel or CSV file → Get Top 5 Students Report**")
    st.sidebar.header("Navigation")
    # Initialize session state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = None

    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("🔧 Controls")
        handle_file_upload()

        if st.session_state.uploaded_data is not None:
            handle_column_mapping()

    # Main content area
    if st.session_state.uploaded_data is not None:
        display_data_preview()

        if st.session_state.column_mapping is not None:
            process_and_display_results()


def handle_file_upload():
    """Handle file upload in sidebar"""
    st.subheader(" Upload File")
    st.markdown("*Supported formats: .xlsx, .xls, .csv (Max: 10MB)*")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=False,
        help="Upload an Excel or CSV file containing student names, roll numbers, and subject marks"
    )

    if uploaded_file is not None:
        # File size check
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
            st.error(" File size exceeds 10MB limit. Please upload a smaller file.")
            return

        try:
            # Determine file type and load accordingly
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            with st.spinner(f" Reading {file_extension.upper()} file..."):
                if file_extension == 'csv':
                    file_data = load_csv_file(uploaded_file)
                else:  # xlsx or xls
                    file_data = load_excel_file(uploaded_file)

            if file_data:
                st.session_state.uploaded_data = file_data
                st.success(" File uploaded successfully!")

                # Display sheet selection if multiple sheets (Excel only)
                if file_data['file_type'] == 'excel' and len(file_data['sheets']) > 1:
                    sheet_names = list(file_data['sheets'].keys())
                    selected_sheet = st.selectbox(
                        " Select worksheet:",
                        sheet_names,
                        help="Choose the worksheet containing student marks"
                    )
                    file_data['selected_sheet'] = selected_sheet
                    st.session_state.uploaded_data = file_data
                else:
                    if file_data['file_type'] == 'excel':
                        file_data['selected_sheet'] = list(file_data['sheets'].keys())[0]
                    st.session_state.uploaded_data = file_data

        except Exception as e:
            st.error(f" Error reading file: {str(e)}")
            st.session_state.uploaded_data = None


def load_csv_file(uploaded_file) -> Optional[Dict]:
    """Load CSV file and return data dictionary"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding=encoding, dtype=str)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("Could not read CSV file with any supported encoding")
        
        # For CSV, we create a single "sheet" structure to maintain compatibility
        sheets = {'Sheet1': df}
        
        return {
            'sheets': sheets,
            'filename': uploaded_file.name,
            'selected_sheet': 'Sheet1',
            'file_type': 'csv',
            'encoding': used_encoding
        }

    except Exception as e:
        raise Exception(f"Failed to read CSV file: {str(e)}")


def load_excel_file(uploaded_file) -> Optional[Dict]:
    """Load Excel file and return data dictionary"""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(uploaded_file)
        sheets = {}

        for sheet_name in excel_file.sheet_names:
            # Read sheet with all data types as object to preserve mixed types
            df = pd.read_excel(excel_file, sheet_name=sheet_name, dtype=str)
            sheets[sheet_name] = df

        return {
            'sheets': sheets,
            'filename': uploaded_file.name,
            'selected_sheet': None,
            'file_type': 'excel'
        }

    except Exception as e:
        raise Exception(f"Failed to read Excel file: {str(e)}")


def handle_column_mapping():
    """Handle column mapping in sidebar"""
    if st.session_state.uploaded_data is None:
        return

    file_data = st.session_state.uploaded_data
    
    # Get the appropriate dataframe based on file type
    if file_data['file_type'] == 'csv':
        df = file_data['sheets']['Sheet1']
    else:  # Excel
        if file_data['selected_sheet'] is None:
            return
        df = file_data['sheets'][file_data['selected_sheet']]

    if df.empty:
        st.warning(" Selected file/worksheet is empty.")
        return

    st.subheader(" Column Mapping")
    st.markdown("*Confirm or adjust column assignments*")

    # Get column names
    columns = df.columns.tolist()

    # Auto-detect columns
    auto_mapping = auto_detect_columns(columns)

    # Name column selection
    name_col = st.selectbox(
        "👤 Name Column:",
        options=columns,
        index=columns.index(auto_mapping['name']) if auto_mapping['name'] in columns else 0,
        help="Column containing student names"
    )

    # Roll number column selection
    roll_col = st.selectbox(
        " Roll Number Column:",
        options=columns,
        index=columns.index(auto_mapping['roll']) if auto_mapping['roll'] in columns else (1 if len(columns) > 1 else 0),
        help="Column containing roll numbers or student IDs"
    )

    # Subject columns selection
    remaining_columns = [col for col in columns if col not in [name_col, roll_col]]
    numeric_columns = get_numeric_columns(df, remaining_columns)

    if not numeric_columns:
        st.warning(" No numeric columns found for subject marks. Please ensure your data contains numeric values.")
        return

    subject_cols = st.multiselect(
        " Subject Columns:",
        options=remaining_columns,
        default=numeric_columns,
        help="Select columns containing subject marks (numeric values)"
    )

    if not subject_cols:
        st.warning(" Please select at least one subject column.")
        return

    # Store column mapping
    st.session_state.column_mapping = {
        'name': name_col,
        'roll': roll_col,
        'subjects': subject_cols
    }

    # Pass percentage threshold
    pass_threshold = st.slider(
        " Pass Threshold (Average %):",
        min_value=0,
        max_value=100,
        value=40,
        help="Average percentage required to be considered as 'passed'"
    )
    st.session_state.column_mapping['pass_threshold'] = pass_threshold


def auto_detect_columns(columns: List[str]) -> Dict[str, str]:
    """Auto-detect likely name and roll columns"""
    name_keywords = ['name', 'student', 'naam', 'nombre']
    roll_keywords = ['roll', 'id', 'number', 'no', 'num', 'reg']

    name_col = None
    roll_col = None

    for col in columns:
        col_lower = col.lower().strip()

        # Check for name column
        if name_col is None:
            for keyword in name_keywords:
                if keyword in col_lower:
                    name_col = col
                    break

        # Check for roll column
        if roll_col is None:
            for keyword in roll_keywords:
                if keyword in col_lower:
                    roll_col = col
                    break

    # Fallback to first columns if not detected
    if name_col is None and columns:
        name_col = columns[0]
    if roll_col is None and len(columns) > 1:
        roll_col = columns[1]

    return {'name': name_col, 'roll': roll_col}


def get_numeric_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Identify columns that contain numeric data"""
    numeric_cols = []

    for col in columns:
        # Check if column contains mostly numeric values
        try:
            # Try to convert to numeric, counting successful conversions
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            non_null_count = numeric_values.notna().sum()
            total_count = len(df[col].dropna())

            # Consider numeric if >70% of non-empty values are convertible to numbers
            if total_count > 0 and (non_null_count / total_count) >= 0.7:
                numeric_cols.append(col)
        except:
            continue

    return numeric_cols


def display_data_preview():
    """Display data preview in main area"""
    if st.session_state.uploaded_data is None:
        return

    file_data = st.session_state.uploaded_data
    
    # Get the appropriate dataframe based on file type
    if file_data['file_type'] == 'csv':
        df = file_data['sheets']['Sheet1']
        sheet_info = "CSV File"
        if 'encoding' in file_data:
            sheet_info += f" (encoding: {file_data['encoding']})"
    else:  # Excel
        df = file_data['sheets'][file_data['selected_sheet']]
        sheet_info = file_data['selected_sheet']

    st.subheader(" Data Preview")

    # Show basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(" Total Rows", len(df))
    with col2:
        st.metric(" Total Columns", len(df.columns))
    with col3:
        st.metric(" Source", sheet_info)

    # Show first 10 rows
    st.markdown("**First 10 rows:**")
    preview_df = df.head(10).copy()

    # Clean up the preview for better display
    for col in preview_df.columns:
        preview_df[col] = preview_df[col].astype(str)
        preview_df[col] = preview_df[col].str.replace('nan', '')

    st.dataframe(preview_df, use_container_width=True)


def process_and_display_results():
    """Process data and display results"""
    if st.session_state.uploaded_data is None or st.session_state.column_mapping is None:
        return

    try:
        # Process the data
        with st.spinner(" Processing student marks..."):
            processed_data = process_student_data()

        if processed_data is None:
            return

        st.session_state.processed_data = processed_data

        # Display results
        display_summary_statistics(processed_data)
        display_top5_leaderboard(processed_data)
        display_top5_chart(processed_data)
        provide_download_option(processed_data)

    except Exception as e:
        st.error(f" Error processing data: {str(e)}")


def process_student_data() -> Optional[Dict]:
    """Process student marks and calculate rankings"""
    file_data = st.session_state.uploaded_data
    column_mapping = st.session_state.column_mapping

    # Get the appropriate dataframe based on file type
    if file_data['file_type'] == 'csv':
        df = file_data['sheets']['Sheet1'].copy()
    else:  # Excel
        df = file_data['sheets'][file_data['selected_sheet']].copy()

    # Extract mapped columns
    name_col = column_mapping['name']
    roll_col = column_mapping['roll']
    subject_cols = column_mapping['subjects']
    pass_threshold = column_mapping['pass_threshold']

    # Clean column names
    df.columns = df.columns.str.strip()

    # Create working dataframe with selected columns
    selected_cols = [name_col, roll_col] + subject_cols
    working_df = df[selected_cols].copy()

    # Clean name and roll columns
    working_df[name_col] = working_df[name_col].astype(str).str.strip()
    working_df[roll_col] = working_df[roll_col].astype(str).str.strip()

    # Remove empty rows
    working_df = working_df.dropna(subset=[name_col, roll_col])
    working_df = working_df[working_df[name_col] != '']
    working_df = working_df[working_df[roll_col] != '']

    if working_df.empty:
        st.error(" No valid student records found after cleaning.")
        return None

    # Process subject columns
    data_warnings = 0

    for col in subject_cols:
        # Convert to numeric, non-numeric becomes 0
        original_values = working_df[col].copy()
        working_df[col] = pd.to_numeric(working_df[col], errors='coerce').fillna(0)

        # Count data warnings (non-numeric entries that became 0)
        non_numeric_count = original_values.astype(str).apply(
            lambda x: not str(x).replace('.', '').replace('-', '').isdigit() and x.strip() != ''
        ).sum()
        data_warnings += non_numeric_count

    # Calculate totals and averages
    working_df['Total'] = working_df[subject_cols].sum(axis=1)
    working_df['Average'] = working_df[subject_cols].mean(axis=1).round(2)

    # Calculate rankings
    # Primary: Total (descending), Secondary: Average (descending), Tertiary: Name (ascending)
    working_df['Rank'] = working_df['Total'].rank(method='min', ascending=False).astype(int)

    # Handle ties with secondary criteria
    ties_mask = working_df.duplicated(subset=['Total'], keep=False)
    if ties_mask.any():
        tied_groups = working_df[ties_mask].groupby('Total')
        for total_score, group in tied_groups:
            # Sort by Average (desc) then Name (asc)
            group_sorted = group.sort_values(['Average', name_col], ascending=[False, True])
            ranks = range(int(group['Rank'].min()), int(group['Rank'].min()) + len(group))
            working_df.loc[group_sorted.index, 'Rank'] = ranks

    # Sort by rank
    working_df = working_df.sort_values('Rank')

    # Calculate summary statistics
    total_students = len(working_df)
    class_avg_total = working_df['Total'].mean()
    class_avg_percentage = working_df['Average'].mean()
    max_total = working_df['Total'].max()
    min_total = working_df['Total'].min()
    pass_count = (working_df['Average'] >= pass_threshold).sum()
    pass_percentage = (pass_count / total_students * 100) if total_students > 0 else 0

    # Get top 5 (or all if fewer than 5)
    top5_df = working_df.head(5)

    return {
        'full_data': working_df,
        'top5_data': top5_df,
        'summary': {
            'total_students': total_students,
            'num_subjects': len(subject_cols),
            'class_avg_total': class_avg_total,
            'class_avg_percentage': class_avg_percentage,
            'max_total': max_total,
            'min_total': min_total,
            'pass_count': pass_count,
            'pass_percentage': pass_percentage,
            'pass_threshold': pass_threshold,
            'data_warnings': data_warnings
        },
        'column_mapping': column_mapping,
        'subject_cols': subject_cols,
        'name_col': name_col,
        'roll_col': roll_col
    }


def display_summary_statistics(processed_data: Dict):
    """Display summary statistics"""
    st.subheader(" Class Summary")

    summary = processed_data['summary']

    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Total Students", 
            summary['total_students']
        )

    with col2:
        st.metric(
            " Class Average", 
            f"{summary['class_avg_percentage']:.1f}%"
        )

    with col3:
        st.metric(
            " Highest Total", 
            f"{summary['max_total']:.0f}"
        )

    with col4:
        st.metric(
            " Pass Rate", 
            f"{summary['pass_percentage']:.1f}%",
            help=f"Students with average ≥ {summary['pass_threshold']}%"
        )

    # Additional details
    if summary['data_warnings'] > 0:
        st.warning(f" {summary['data_warnings']} non-numeric entries were treated as 0")


def display_top5_leaderboard(processed_data: Dict):
    """Display top 5 students leaderboard"""
    st.subheader(" Top 5 Students Leaderboard")

    top5_df = processed_data['top5_data'].copy()
    name_col = processed_data['name_col']
    roll_col = processed_data['roll_col']
    subject_cols = processed_data['subject_cols']

    # Prepare display dataframe
    display_df = top5_df[[name_col, roll_col] + subject_cols + ['Total', 'Average', 'Rank']].copy()

    # Format numeric columns
    for col in subject_cols + ['Total']:
        display_df[col] = display_df[col].astype(int)

    # Add medal emojis for top 3
    rank_medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    display_df['Medal'] = display_df['Rank'].map(rank_medals).fillna("")

    # Reorder columns to show medal first
    cols = ['Medal', name_col, roll_col] + subject_cols + ['Total', 'Average', 'Rank']
    display_df = display_df[cols]

    # Style the dataframe
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Medal": st.column_config.TextColumn("🏅", width="small"),
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Average": st.column_config.NumberColumn("Average", format="%.2f")
        }
    )


def display_top5_chart(processed_data: Dict):
    """Display bar chart of top 5 total scores"""
    st.subheader(" Top 5 Total Scores")

    top5_df = processed_data['top5_data'].copy()
    name_col = processed_data['name_col']

    # Create bar chart
    fig = px.bar(
        top5_df,
        x=name_col,
        y='Total',
        title="Top 5 Students - Total Marks",
        color='Total',
        color_continuous_scale='viridis',
        text='Total'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        xaxis_title="Student Name",
        yaxis_title="Total Marks",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def provide_download_option(processed_data: Dict):
    """Provide download option for Excel report"""
    st.subheader(" Download Report")

    try:
        # Generate Excel file
        excel_buffer = create_excel_report(processed_data)

        # Create download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"top5_students_{timestamp}.xlsx"

        st.download_button(
            label=" Download Top 5 Students Report (.xlsx)",
            data=excel_buffer,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download Excel file with Top 5 students and summary statistics"
        )

        st.success(" Excel report generated successfully!")

    except Exception as e:
        st.error(f" Error generating Excel report: {str(e)}")


def create_excel_report(processed_data: Dict) -> bytes:
    """Create Excel report with Top 5 and Summary sheets"""
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Get data
        top5_df = processed_data['top5_data'].copy()
        summary = processed_data['summary']
        name_col = processed_data['name_col']
        roll_col = processed_data['roll_col']
        subject_cols = processed_data['subject_cols']

        # Prepare Top 5 sheet
        top5_export = top5_df[[name_col, roll_col] + subject_cols + ['Total', 'Average', 'Rank']].copy()

        # Format data for export
        for col in subject_cols + ['Total']:
            top5_export[col] = top5_export[col].astype(int)

        # Write Top 5 sheet
        top5_export.to_excel(writer, sheet_name='Top 5', index=False)

        # Create Summary sheet
        summary_data = [
            ['Metric', 'Value'],
            ['Total Students', summary['total_students']],
            ['Number of Subjects', summary['num_subjects']],
            ['Class Average Total', f"{summary['class_avg_total']:.2f}"],
            ['Class Average Percentage', f"{summary['class_avg_percentage']:.2f}%"],
            ['Highest Total Score', summary['max_total']],
            ['Lowest Total Score', summary['min_total']],
            ['Students Passed', summary['pass_count']],
            ['Pass Percentage', f"{summary['pass_percentage']:.2f}%"],
            ['Pass Threshold', f"{summary['pass_threshold']}%"],
            ['Data Warnings (Non-numeric entries treated as 0)', summary['data_warnings']],
            ['Report Generated', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]

        summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Get workbook and worksheets for formatting
        workbook = writer.book

        # Format Top 5 sheet
        top5_worksheet = writer.sheets['Top 5']

        # Header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        # Write headers with formatting
        for col_num, value in enumerate(top5_export.columns.values):
            top5_worksheet.write(0, col_num, value, header_format)

        # Auto-adjust column widths
        for i, col in enumerate(top5_export.columns):
            max_length = max(
                top5_export[col].astype(str).map(len).max(),
                len(str(col))
            )
            top5_worksheet.set_column(i, i, max_length + 2)

        # Format Summary sheet
        summary_worksheet = writer.sheets['Summary']

        # Write headers with formatting
        for col_num, value in enumerate(summary_df.columns.values):
            summary_worksheet.write(0, col_num, value, header_format)

        # Auto-adjust column widths for summary
        summary_worksheet.set_column(0, 0, 40)  # Metric column
        summary_worksheet.set_column(1, 1, 20)  # Value column

    buffer.seek(0)
    return buffer.getvalue()


if __name__ == "__main__":
    main()