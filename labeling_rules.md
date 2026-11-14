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
- Kitchen appliance → Shopping
- Payment to a person without a stated purpose → Other
- Multiple categories without a clear main purpose → Other
- Use only the information provided; do not guess missing details.
