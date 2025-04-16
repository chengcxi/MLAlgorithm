import csv
from datetime import datetime

def convert_date_format(input_file, output_file, date_column):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                # Convert the date format
                old_date = row[date_column]
                new_date = datetime.strptime(old_date, "%m/%d/%Y").strftime("%Y-%m-%d")
                row[date_column] = new_date
            except ValueError:
                print(f"Skipping invalid date: {row[date_column]}")
            writer.writerow(row)

# Example usage
convert_date_format('newertqqq.csv', 'uptodatetqqq.csv', 'Date')
