# Expense Labeling Rules

Classify expenses by what was purchased, not just the store name.

| Category      | Includes                                                    |
| ------------- | ----------------------------------------------------------- |
| Food          | Meals, drinks, and groceries                                |
| Transport     | Taxi, public transport, fuel, and parking                   |
| Bills         | Electricity, water, internet, and phone plans               |
| Shopping      | Clothes, electronics, furniture, and household goods        |
| Health        | Medical appointments, medicine, and dental care             |
| Entertainment | Cinema, games, and entertainment subscriptions              |
| Other         | Expenses outside the categories or with insufficient detail |

## Ambiguous cases

- Netflix subscription → Entertainment
- Internet subscription → Bills
- Supermarket groceries → Food
- Airport parking, an airport shuttle, or a taxi to a hotel → Transport
- A meal at an airport → Food. A hotel room near an airport → Other.
- Kitchen appliance → Shopping
- Payment to a person without a stated purpose → Other
- Multiple categories without a clear main purpose → Other
- Use only the information provided; do not guess missing details.

## Adding everyday descriptions

1. Write a short expense in the words a person might type. Record its purpose and assign one of the seven labels above. Include purchases from unfamiliar merchants and descriptions without a merchant.
2. Add the description to `data/everyday_expenses.csv` with its target category and partition. Keep variants of the same purchase or template in one partition. The current rows are authored English examples.
3. Use the training rows to fit the model and the validation rows to compare changes. Keep the original demo examples and audit probes separate. For a future release, collect a fresh, separately labeled test set before further tuning.
4. Run `01_data_inspection.ipynb` followed by `notebooks/02_final_project.ipynb.ipynb` to rebuild from raw transactions and the authored CSV. From the repository root, run `python tests/audit_classifier.py` to check both sources and the Streamlit interface.
