# Student-Grading-With-AI
A powerful Streamlit web application for analyzing student marks from Excel files and generating comprehensive top 5 student reports.

## 📊 Features

### Core Functionality
- **Excel File Upload**: Supports both `.xlsx` and `.xls` formats (up to 10MB)
- **Multi-sheet Support**: Select from multiple worksheets if available
- **Smart Column Detection**: Automatically detects Name, Roll Number, and Subject columns
- **Interactive Column Mapping**: Override auto-detection with intuitive UI controls
- **Data Cleaning**: Handles missing data, non-numeric entries, and whitespace
- **Advanced Ranking**: Multi-tier ranking system with intelligent tie-breaking

### Analysis & Visualization
- **Class Summary Statistics**: Average scores, pass rates, and distribution metrics  
- **Top 5 Leaderboard**: Interactive table with medal indicators and detailed breakdown
- **Visual Charts**: Beautiful bar charts showing top performer comparison
- **Data Quality Warnings**: Alerts for non-numeric entries and data issues

### Export & Download
- **Professional Excel Reports**: Two-sheet workbook with Top 5 and Summary data
- **Formatted Output**: Clean headers, proper numeric formatting, and styled sheets
- **In-Memory Processing**: No server-side file storage for security and performance

## 📋 How to Use

### Step 1: Upload Your Excel File
1. Click "Browse files" in the sidebar
2. Select your Excel file (.xlsx or .xls)
3. If multiple sheets exist, choose the one containing student data
4. Review the data preview to ensure correct upload

### Step 2: Map Your Columns
1. **Name Column**: Select the column containing student names
2. **Roll Number Column**: Choose the column with student IDs/roll numbers  
3. **Subject Columns**: Select all columns containing numeric subject marks
4. **Pass Threshold**: Set the average percentage required to "pass" (default: 40%)

### Step 3: Analyze Results
- **Class Summary**: View overall statistics and performance metrics
- **Top 5 Leaderboard**: See the highest-performing students with medals
- **Visual Chart**: Compare top 5 student totals in an interactive bar chart

### Step 4: Download Report
- Click "Download Top 5 Students Report" 
- Get a professionally formatted Excel file with two sheets:
  - **Top 5 Sheet**: Complete details of top performers
  - **Summary Sheet**: Class statistics and metadata

## 📊 Data Requirements

### Expected Format
Your Excel file should contain:
- **Student Names**: Text column (e.g., "Name", "Student Name")
- **Roll Numbers**: Text/numeric column (e.g., "Roll No", "ID", "Registration")  
- **Subject Marks**: Numeric columns with subject names as headers

### Example Structure
| Name | Roll No | Math | Science | English | Computer |
|------|---------|------|---------|---------|----------|
| Alice Johnson | ST001 | 85 | 92 | 78 | 88 |
| Bob Smith | ST002 | 76 | 84 | 82 | 79 |

### Data Handling
- **Missing Values**: Treated as 0 (with warning counter)
- **Non-numeric Entries**: Converted to 0 (e.g., "Absent", "N/A")
- **Empty Rows**: Automatically filtered out
- **Whitespace**: Trimmed from names and roll numbers

## 🏆 Ranking System

The application uses a sophisticated multi-tier ranking system:

1. **Primary**: Total marks (descending)
2. **Tie-breaker 1**: Average percentage (descending)  
3. **Tie-breaker 2**: Student name (alphabetical)

This ensures consistent, fair rankings even with identical scores.
