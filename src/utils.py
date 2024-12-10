import streamlit as st

def sidebar_markdown(company_details):
    company_details["description"] = clean_text(company_details["description"])
    markdown_content = f"""
    <style>
        .sidebar-container {{
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #fff;
        }}
        .sidebar-header {{
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }}
        .sidebar-section {{
            margin-top: 20px;
        }}
        .sidebar-section-title {{
            font-size: 16px;
            font-weight: bold;
            color: #2E86C1;
            margin-bottom: 8px;
        }}
        .sidebar-detail {{
            font-size: 14px;
            margin-bottom: 6px;
        }}
    </style>
    
    <div class="sidebar-container">
        <div class="sidebar-header">
            🏢 {company_details['company_name']}
        </div>
        <div class="sidebar-section">
            <div class="sidebar-detail"><b>🌍 Place:</b> {company_details['country']}</div>
            <div class="sidebar-detail"><b>🏢 Industry Category:</b> {company_details['industry_category']}</div>
            <div class="sidebar-detail"><b>📊 Sector:</b> {company_details['sector']}</div>
            <div class="sidebar-detail"><b>🏭 Industry:</b> {company_details['industry']}</div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-title">🛠️ Products and Services:</div>
            <div class="sidebar-detail">{company_details['products_and_services']}</div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-title">📄 Description:</div>
            <div class="sidebar-detail">{company_details['description']}</div>
        </div>
    </div>
    """
    return markdown_content


def clean_text(text):
    """
    Helper function to clean and sanitize text by removing unwanted characters and escape sequences.
    
    Args:
        text (str): The input text to clean.
    
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        text = str(text)  # Ensure the input is a string
    # Remove unwanted escape sequences and whitespace
    text = text.replace("\n", " ").replace("\r", " ").replace("x000D", "").replace("_", "").strip()
    # Replace multiple spaces with a single space
    text = " ".join(text.split())
    return text
