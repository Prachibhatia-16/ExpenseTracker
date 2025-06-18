class Expense:
    def __init__(self, date, description, category, amount, expense_type, receipt):
        self.date = date
        self.description = description
        self.category = category
        self.amount = amount
        self.expense_type = expense_type
        self.receipt = receipt

    def __repr__(self):
        return (f"Expense: {self.description}\n"
                f"Amount: {self.amount}\n"
                f"Category: {self.category}\n"
                f"Receipt: {self.receipt}")

