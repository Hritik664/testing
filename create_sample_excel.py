#!/usr/bin/env python3
"""
Create a sample Excel file for testing bulk question processing.
"""

import pandas as pd
from pathlib import Path

# Sample questions data
questions_data = [
    {
        'Question': 'Revenue guidance for the next few years',
        'Type': 'Figure',
        'Figure in cr': 'cr',
        'Period': '2 years',
        'Figure': '3500'
    },
    {
        'Question': 'Guidance on R&D expense',
        'Type': 'Range',
        'Figure in cr': '%',
        'Period': 'Annual',
        'Figure': '1.9 to 2'
    },
    {
        'Question': 'What is the current capacity utilization',
        'Type': 'Figure',
        'Figure in cr': '%',
        'Period': '',
        'Figure': '6500%'
    },
    {
        'Question': 'What is the outlook on future capex',
        'Type': 'Range',
        'Figure in cr': 'million $',
        'Period': 'Annual',
        'Figure': '8 to 10'
    },
    {
        'Question': 'Guidance on working capital',
        'Type': 'Range',
        'Figure in cr': 'days',
        'Period': 'Annual',
        'Figure': '125 to 135'
    },
    {
        'Question': 'Revenue growth guidance for the next few years',
        'Type': 'Figure',
        'Figure in cr': '%',
        'Period': '2 years',
        'Figure': '17%'
    },
    {
        'Question': 'Guidance on EBITDA margin',
        'Type': 'Range',
        'Figure in cr': '%',
        'Period': '',
        'Figure': '21 to 22%'
    },
    {
        'Question': 'What was the sales volume in Q1FY25',
        'Type': 'Figure',
        'Figure in cr': 'mnT',
        'Period': 'Q1FY25',
        'Figure': '7.4'
    },
    {
        'Question': 'What was the cost on a per ton basis in Q1FY26',
        'Type': 'Figure',
        'Figure in cr': 'per ton',
        'Period': 'Q1FY26',
        'Figure': '3932'
    },
    {
        'Question': 'What was the total cost in Rs Cr in Q1FY26',
        'Type': 'Figure',
        'Figure in cr': 'Rs Cr',
        'Period': 'Q1FY26',
        'Figure': '2753'
    },
    {
        'Question': 'What is the current capacity',
        'Type': 'Figure',
        'Figure in cr': 'mnT',
        'Period': 'Current',
        'Figure': '49.5'
    },
    {
        'Question': 'What is capex for new expansion',
        'Type': 'Figure',
        'Figure in cr': 'cr',
        'Period': 'New expansion',
        'Figure': '3287'
    },
    {
        'Question': 'What was the revenue growth in Q1FY26 in USD terms',
        'Type': 'Figure',
        'Figure in cr': '%',
        'Period': 'Q1FY26',
        'Figure': '-1.1%'
    },
    {
        'Question': 'What is the company guidance on capex for the year',
        'Type': 'Text',
        'Figure in cr': '',
        'Period': 'Annual',
        'Figure': 'No specific guidance'
    },
    {
        'Question': 'What caused the jump in employee costs over the last few quarters',
        'Type': 'List',
        'Figure in cr': '',
        'Period': 'Last few quarters',
        'Figure': 'Adding people, higher QVA, tactical interventions'
    }
]

def create_sample_excel():
    """Create a sample Excel file with questions."""
    
    # Create DataFrame
    df = pd.DataFrame(questions_data)
    
    # Create examples directory if it doesn't exist
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Save to Excel
    excel_path = examples_dir / "sample_questions.xlsx"
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    print(f"✅ Created sample Excel file: {excel_path}")
    print(f"📊 Contains {len(questions_data)} questions")
    print("\n📋 Sample questions:")
    for i, row in df.head(3).iterrows():
        print(f"  {i+1}. {row['Question']}")
    
    print(f"\n📁 File saved to: {excel_path.absolute()}")
    print("💡 You can now use this file to test bulk question processing!")

if __name__ == "__main__":
    create_sample_excel() 