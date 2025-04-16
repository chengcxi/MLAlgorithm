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
                new_date = None
                for fmt in ("%m/%d/%Y", "%b%d-%Y"):
                    try:
                        new_date = datetime.strptime(old_date, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                if new_date:
                    row[date_column] = new_date
                else:
                    print(f"Skipping invalid date: {row[date_column]}")
            except Exception as e:
                print(f"Error processing row: {e}")
            writer.writerow(row)

# Example usage
convert_date_format('vix.csv', 'fixvix.csv', 'Date')
