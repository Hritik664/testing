#!/usr/bin/env python3
"""
Create properly formatted Excel file for bulk question processing
"""

import pandas as pd
import os

# Create sample data with the correct column structure
data = {
    'Question': [
        'Revenue guidance for the next few years',
        'Guidance on R&D expense',
        'What is the current capacity utilization',
        'What is the outlook on future capex',
        'Guidance on working capital',
        'Revenue growth guidance for the next few years',
        'Guidance on EBITDA margin',
        'What was the sales volume in Q1FY25',
        'What was the cost on a per ton basis in Q1FY26',
        'What was the total cost in Rs Cr in Q1FY26',
        'What is the current capacity',
        'What is capex for new expansion',
        'What was the revenue growth in Q1FY26 in USD terms',
        'What is the company guidance on capex for the year',
        'What caused the jump in employee costs over the last few quarters'
    ],
    'Type': [
        'Figure',
        'Range',
        'Figure',
        'Range',
        'Range',
        'Figure',
        'Range',
        'Figure',
        'Figure',
        'Figure',
        'Figure',
        'Figure',
        'Figure',
        'Text',
        'Text'
    ],
    'Figure in cr': [
        'cr',
        '%',
        '%',
        'million $',
        'days',
        '%',
        '%',
        'mnT',
        'Rs',
        'cr',
        'mnT',
        'cr',
        '%',
        '',
        ''
    ],
    'Period': [
        '2 years',
        'Annual',
        'Current',
        'Annual',
        'Annual',
        '2 years',
        'Current',
        'Q1FY25',
        'Q1FY26',
        'Q1FY26',
        'Current',
        'New expansion',
        'Q1FY26',
        'FY26',
        'Last few quarters'
    ],
    'Figure': [
        '3500',
        '1.9 to 2',
        '6500%',
        '8 to 10',
        '125 to 135',
        '17%',
        '21 to 22%',
        '7.4',
        '3932',
        '2753',
        '49.5',
        '3287',
        '-1.1%',
        'No specific guidance',
        'Adding people, higher QVA, promotions'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create examples directory if it doesn't exist
os.makedirs('examples', exist_ok=True)

# Save to Excel
output_file = 'examples/Q&A.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')

print("✅ Created Q&A.xlsx with correct column structure")
print("📊 Contains 15 financial questions")
print("📋 Columns: Question, Type, Figure in cr, Period, Figure")
print(f"📁 File saved to: {output_file}")
print("\n💡 You can now upload this file in the 'Bulk Questions' tab!") 