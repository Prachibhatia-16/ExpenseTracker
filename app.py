from flask import Flask, render_template, request, make_response, redirect, url_for, flash, send_file, jsonify, session
from flask_bcrypt import Bcrypt
import re
import json
import traceback
from functools import wraps
from expense_class import Expense
from datetime import datetime
import mysql.connector
from mysql.connector.types import RowType
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from decimal import Decimal
import os
import requests
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from weasyprint import HTML
from calendar import month_name
from fpdf import FPDF
from werkzeug.security import generate_password_hash, check_password_hash
from typing import Dict, List, Optional, Union, Any, Sequence, cast, Tuple, TypedDict, TypeVar, Mapping

T = TypeVar('T')

class BudgetRow(TypedDict):
    budget: float
    month: int

app = Flask(__name__)
# Configure Gemini AI
model = None
try:
    import google.generativeai as genai
    genai.configure(api_key='AIzaSyBRDfBOoDQquy3wxGNzVYItFenfaHet358')
    model = genai.GenerativeModel('gemini-2.0-flash')
except ImportError:
    print("Warning: google-generativeai package is not installed")
except Exception as e:
    print(f"Error configuring Gemini AI: {str(e)}")

def gemini_ask(user_input: str) -> str:
    if not model:
        return "Gemini AI is not properly configured. Please ensure the google-generativeai package is installed."
    try:
        response = model.generate_content(user_input)
        return response.text if response and response.text else "No response generated"
    except Exception as e:
        print(f"Error in Gemini AI: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now."

def db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Radha@1981",
            database="Expense_tracker"
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        raise

def create_tables():
    connection = db_connection()
    cursor = connection.cursor()
    try:
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create expenses_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                date DATE NOT NULL,
                description VARCHAR(255) NOT NULL,
                category VARCHAR(50) NOT NULL,
                amount DECIMAL(10, 2) NOT NULL,
                expense_type VARCHAR(50) NOT NULL,
                receipt VARCHAR(255) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Create budget_table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS budget_table (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                year INT NOT NULL,
                month INT NOT NULL,
                budget DECIMAL(10, 2) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE KEY unique_budget (user_id, year, month)
            )
        """)

        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error creating tables: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

# Initialize the app and create tables
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a secure secret key
bcrypt = Bcrypt(app)

# Create tables on startup
with app.app_context():
    create_tables()

UPLOAD_FOLDER = 'static/receipts'  # or any folder you want to save images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB max upload size
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    flash('File is too large. Maximum allowed size is 2 MB.', 'danger')
    return redirect(request.url)

category_map = {
    'food & drink': 'Food',
    'snacks': 'Food',
    'hunger': 'Food',
    'food': 'Food',
    'utilities': 'Utilities',
    'electricity': 'Utilities',
    'shopping': 'Shopping',
    'groceries': 'Shopping',
    'rent': 'Rent',
    'investment': 'Income',
    'salary': 'Income',
    'other': 'Other',
    'entertainment': 'Entertainment',
}

def normalize_category(cat):
    return category_map.get(cat.lower().strip(), 'Other')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def load_expenses(year: Optional[int] = None, month: Optional[int] = None, 
                 category: Optional[str] = None, search: Optional[str] = None, 
                 user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    if not user_id:
        return []

    try:
        connection = db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = "SELECT * FROM expenses_data WHERE user_id = %s"
        params: List[Union[int, str]] = [user_id]

        if year:
            query += " AND YEAR(date) = %s"
            params.append(year)
        if month:
            query += " AND MONTH(date) = %s"
            params.append(month)
        if category:
            query += " AND category = %s"
            params.append(category)
        if search:
            query += " AND description LIKE %s"
            params.append(f"%{search}%")

        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        expenses = cursor.fetchall()
        
        return cast(List[Dict[str, Any]], expenses)
    except mysql.connector.Error as err:
        print(f"Error loading expenses: {err}")
        return []
    finally:
        cursor.close()
        connection.close()

def get_expenses_by_filters(year: int, month: Optional[int] = None, 
                          search_desc: str = '', filter_category: str = '', 
                          start_date: str = '', end_date: str = '') -> List[Dict[str, Any]]:
    conn = db_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT * FROM expenses_data WHERE YEAR(date) = %s"
    params: List[Union[int, str]] = [year]

    if month:
        query += " AND MONTH(date) = %s"
        params.append(month)

    if search_desc:
        query += " AND description LIKE %s"
        params.append('%' + search_desc + '%')

    if filter_category:
        query += " AND category = %s"
        params.append(filter_category)

    if start_date:
        query += " AND date >= %s"
        params.append(start_date)

    if end_date:
        query += " AND date <= %s"
        params.append(end_date)

    query += " ORDER BY date DESC"

    cursor.execute(query, tuple(params))
    expenses = cursor.fetchall()
    conn.close()
    return cast(List[Dict[str, Any]], expenses)

def safe_get(row: Optional[Dict[str, Any]], key: str, default: T = None) -> T:
    """Safely get a value from a dictionary with a default."""
    if row is None:
        return default
    return row[key] if key in row else default

def get_budget(year: int, month: Optional[int] = None, user_id: Optional[int] = None) -> float:
    if not user_id:
        return 0.0
    try:
        connection = db_connection()
        cursor = connection.cursor(dictionary=True)
        if month:
            cursor.execute("""
                SELECT budget FROM budget_table
                WHERE user_id = %s AND year = %s AND month = %s
            """, (user_id, year, month))
            row = cast(Optional[Dict[str, Any]], cursor.fetchone())
            budget_value = safe_get(row, 'budget') if row else None
            return float(budget_value) if budget_value is not None else 0.0
        else:
            cursor.execute("""
                SELECT SUM(budget) as total_budget FROM budget_table
                WHERE user_id = %s AND year = %s
            """, (user_id, year))
            row = cast(Optional[Dict[str, Any]], cursor.fetchone())
            total_budget = safe_get(row, 'total_budget') if row else None
            return float(total_budget) if total_budget is not None else 0.0
    except mysql.connector.Error as err:
        print(f"Error getting budget: {err}")
        return 0.0
    finally:
        cursor.close()
        connection.close()

def save_expense(expense_data: Dict[str, Any]) -> int:
    try:
        connection = db_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            INSERT INTO expenses_data 
            (user_id, date, description, category, amount, expense_type, receipt) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            expense_data['user_id'],
            expense_data['date'],
            expense_data['description'],
            expense_data['category'],
            expense_data['amount'],
            expense_data['expense_type'],
            expense_data.get('receipt')
        ))
        
        connection.commit()
        last_id = cursor.lastrowid
        if last_id is None:
            raise ValueError("Failed to get last inserted ID")
        return last_id
    except mysql.connector.Error as err:
        print(f"Error saving expense: {err}")
        raise
    finally:
        cursor.close()
        connection.close()

def summarize_expenses(expenses):
    summary = {}
    total = Decimal('0.0')
    for e in expenses:
        amt = Decimal(str(e.amount))
        summary[e.category] = summary.get(e.category, Decimal('0.0')) + amt
        total += amt
    return summary, total
def get_summary_by_category(expenses):
    summary = {}
    for expense in expenses:
        category = expense['category']
        amount = expense['amount']
        if category in summary:
            summary[category] += amount
        else:
            summary[category] = amount
    return summary

def get_expenses(year, month=None):
    conn = db_connection()
    cursor = conn.cursor(dictionary=True)  # <-- this returns dicts
    if month:
        cursor.execute("""
            SELECT * FROM expenses_data
            WHERE YEAR(date) = %s AND MONTH(date) = %s
        """, (year, month))
    else:
        cursor.execute("""
            SELECT * FROM expenses_data
            WHERE YEAR(date) = %s
        """, (year,))
    expenses = cursor.fetchall()
    conn.close()
    return expenses

def update_budget(budget: float, year: int, month: Optional[int] = None, user_id: Optional[int] = None) -> bool:
    if not user_id:
        return False

    try:
        connection = db_connection()
        cursor = connection.cursor()

        if month:
            cursor.execute("""
                INSERT INTO budget_table (user_id, year, month, budget)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE budget = %s
            """, (user_id, year, month, budget, budget))
        else:
            # Update all months for the year
            for m in range(1, 13):
                cursor.execute("""
                    INSERT INTO budget_table (user_id, year, month, budget)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE budget = %s
                """, (user_id, year, m, budget, budget))

        connection.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Error updating budget: {err}")
        return False
    finally:
        cursor.close()
        connection.close()

def get_yearly_budget_dict(year: int) -> Dict[int, float]:
    conn = db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT month, budget FROM budget_table WHERE user_id=%s AND year=%s
        """, (1, year))
        result = cursor.fetchall()
        return {int(cast(dict, row)['month']): float(cast(dict, row)['budget']) for row in result} if result else {}
    finally:
        cursor.close()
        conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        connection = db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id, password FROM users WHERE username = %s", (username,))
        user = cast(dict, cursor.fetchone())
        cursor.close()
        connection.close()
        if user and check_password_hash(str(user.get('password')), password):
            session['user_id'] = user.get('id')
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']  # New email field
        password = request.form['password']
        
        if not email:
            flash('Email is required.', 'danger')
            return render_template('signup.html')

        hashed_password = generate_password_hash(password)
        connection = db_connection()
        cursor = connection.cursor()

        # Check if username or email already exists
        cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
        if cursor.fetchone():
            flash('Username or Email already exists.', 'danger')
        else:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, hashed_password)
            )
            connection.commit()
            flash('Signup successful! Please log in.', 'success')
            cursor.close()
            connection.close()
            return redirect(url_for('login'))

        cursor.close()
        connection.close()
    return render_template('signup.html')

@app.route('/')
@login_required
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    current_year = datetime.now().year
    selected_year = request.args.get('year', session.get('selected_year', current_year), type=int)
    selected_month = request.args.get('month', session.get('selected_month'), type=int)
    search = request.args.get('search', '')
    category = request.args.get('category', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    session['selected_year'] = selected_year
    session['selected_month'] = selected_month
    expenses = get_expenses_by_filters(selected_year, selected_month, search, category, start_date, end_date)
    budget = get_budget(selected_year, selected_month, session['user_id'])
    categories = sorted(set(category_map.values()))
    total_spent = sum(Decimal(str(exp['amount'])) for exp in expenses) if expenses else Decimal('0.0')
    remaining = Decimal(str(budget)) - total_spent if budget is not None else Decimal('0.0')
    summary = {}
    for exp in expenses:
        cat = exp['category']
        amount = exp['amount']
        summary[cat] = summary.get(cat, 0) + amount
    return render_template('home.html',
                           current_year=current_year,
                           selected_year=selected_year,
                           selected_month=selected_month,
                           search=search,
                           category=category,
                           start_date=start_date,
                           end_date=end_date,
                           expenses=expenses,
                           budget=budget,
                           total_spent=total_spent,
                           remaining=remaining,
                           summary=summary,
                           categories=categories)

@app.route('/set_budget', methods=['GET', 'POST'])
@login_required
def set_budget():
    # Step 1: Check for chatbot-driven auto_budget
    auto_data = session.pop('auto_budget', None)  # Use only once

    try:
        if auto_data:
            # Handle chatbot data
            budget = float(auto_data['amount'])
            month = auto_data.get('month')
            year = datetime.now().year  # You can also parse year from chatbot if added

            # Store selected month and year in session (if needed elsewhere)
            session['selected_month'] = month
            session['selected_year'] = year

            update_budget(budget, year, month)
            flash(f"Budget for {month} set to ₹{budget}.", "success")
            return redirect(url_for('home'))

        elif request.method == 'POST':
            # Handle normal form data
            budget = float(request.form['budget'])
            year = session.get('selected_year', datetime.now().year)
            month = session.get('selected_month')  # Could be None

            update_budget(budget, year, month)
            flash("Budget set successfully!", "success")
            return redirect(url_for('home'))

    except (ValueError, KeyError) as e:
        flash("Invalid input. Please check your entries.", "danger")

    # GET request or error case
    current_year = datetime.now().year
    selected_year = session.get('selected_year', current_year)
    selected_month = session.get('selected_month')

    return render_template('set_budget.html',
                           current_year=current_year,
                           selected_year=selected_year,
                           selected_month=selected_month)

from werkzeug.utils import secure_filename
import os

@app.route('/add_expense', methods=['GET', 'POST'])
@login_required
def add_expense():
    categories = sorted(set(category_map.values()))
    user_id = session['user_id']
    auto_data = session.pop('auto_add', None)
    receipt_filename = None

    if request.method == 'POST' or auto_data:
        if auto_data:
            # Handle auto data from chatbot
            date_input = auto_data.get('date', '')
            if date_input == 'today':
                date = datetime.today().strftime('%Y-%m-%d')
            else:
                date = date_input

            if not date:
                flash("Please provide a valid date for the expense.", "danger")
                return redirect(url_for('add_expense'))
            try:
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                flash("Invalid date format. Please use YYYY-MM-DD.", "danger")
                return redirect(url_for('add_expense'))

            description = "Auto entry"
            category = auto_data.get('category', 'Others').capitalize()
            amount = float(auto_data.get('amount', 0))
            expense_type = 'debit'
        else:
            # Handle manual form
            date = request.form.get('date', '').strip()
            if not date:
                flash("Please provide a valid date for the expense.", "danger")
                return redirect(url_for('add_expense'))
            try:
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                flash("Invalid date format. Please use YYYY-MM-DD.", "danger")
                return redirect(url_for('add_expense'))

            description = request.form.get('description', '').strip()
            category = request.form.get('category', 'Others').strip()
            amount = float(request.form.get('amount', 0))
            expense_type = request.form.get('expense_type', 'debit')

            # Handle receipt file upload
            if 'receipt' in request.files:
                receipt = request.files['receipt']
                if receipt and receipt.filename != '':
                    receipt_filename = secure_filename(receipt.filename)
                    receipt_path = os.path.join('static', 'receipts', receipt_filename)
                    receipt.save(receipt_path)

        # Insert into database
        connection = db_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO expenses_data (date, description, category, amount, expense_type, user_id, receipt)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (date, description, category, amount, expense_type, user_id, receipt_filename))
            connection.commit()
            flash("Expense added successfully.", "success")
            return redirect(url_for('home'))
        except Exception as e:
            flash(f"Failed to add expense: {str(e)}", "danger")
        finally:
            cursor.close()
            connection.close()

    return render_template('add_expense.html', categories=categories)


@app.template_filter('month_name')
def month_name_filter(month_num):
    if month_num is None:
        return ''
    return month_name[month_num]

@app.route('/delete_expense/<int:expense_id>', methods=['GET', 'POST'])
@login_required
def delete_expense(expense_id):
    user_id = session.get('user_id')
    if not user_id:
        flash("Unauthorized access. Please login.", "danger")
        return redirect(url_for('login'))

    connection = db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM expenses_data WHERE id = %s AND user_id = %s", (expense_id, user_id))
        if cursor.rowcount == 0:
            flash("Unauthorized or non-existent expense.", "danger")
        else:
            connection.commit()
            flash("Expense deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting expense: {str(e)}", "danger")
    finally:
        cursor.close()
        connection.close()

    return redirect(url_for('home'))

@app.route('/export_pdf')
@login_required
def export_pdf():
    selected_year = request.args.get('year', type=int)
    if selected_year is None:
        selected_year = datetime.now().year
    selected_month = request.args.get('month', type=int)
    category = request.args.get('category', default=None, type=str)
    search = request.args.get('search', default=None, type=str)
    user_id = session.get('user_id')
    expenses = load_expenses(
        year=selected_year,
        month=selected_month,
        category=category,
        search=search,
        user_id=user_id
    )
    for e in expenses:
        if isinstance(e['date'], str):
            e['date'] = datetime.strptime(e['date'], '%Y-%m-%d')
    
    # Convert all monetary values to Decimal for consistent handling
    summary = {}
    total_spent = Decimal('0')
    for e in expenses:
        amount = Decimal(str(e['amount']))
        category = e['category']
        summary[category] = summary.get(category, Decimal('0')) + amount
        total_spent += amount
    
    # Convert budget to Decimal
    budget = Decimal(str(get_budget(selected_year, selected_month, user_id)))
    remaining = budget - total_spent if budget is not None else None
    
    rendered_html = render_template(
        'pdf_report.html',
        year=selected_year,
        month=selected_month,
        category=category,
        search=search,
        expenses=expenses,
        summary=summary,
        budget=float(budget),  # Convert back to float for template
        total_spent=float(total_spent),  # Convert back to float for template
        remaining=float(remaining) if remaining is not None else None  # Convert back to float for template
    )
    pdf = HTML(string=rendered_html).write_pdf()
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=expenses_{selected_year}_{selected_month or "all"}.pdf'
    return response

@app.route('/export_csv')
@login_required
def export_csv():
    user_id = session.get('user_id')
    if not user_id:
        flash("Unauthorized access. Please login.", "danger")
        return redirect(url_for('login'))
    selected_year = request.args.get('year', type=int)
    if selected_year is None:
        selected_year = datetime.now().year
    selected_month = request.args.get('month', type=int)
    category = request.args.get('category', default=None, type=str)
    search = request.args.get('search', default=None, type=str)
    expenses = load_expenses(
        year=selected_year,
        month=selected_month,
        category=category,
        search=search,
        user_id=user_id
    )
    for e in expenses:
        if isinstance(e['date'], str):
            e['date'] = datetime.strptime(e['date'], '%Y-%m-%d')
    data = [{
        'date': e['date'].strftime('%Y-%m-%d'),
        'description': e['description'],
        'category': e['category'],
        'amount': e['amount'],
        'expense_type': e['expense_type']
    } for e in expenses]
    df = pd.DataFrame(data)
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return send_file(
        csv_buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='expenses_export.csv'
    )

@app.route('/select_chart')
@login_required
def select_chart():
    current_year = datetime.now().year
    return render_template('select_chart.html', current_year=current_year)

@app.route('/summary_chart')
@login_required
def summary_chart():
    year = request.args.get('year', type=int) or session.get('selected_year')
    month = request.args.get('month', default=session.get('selected_month'), type=int)
    chart_type = request.args.get('chart_type', default='pie')
    if year:
        session['selected_year'] = year
    if month is not None:
        session['selected_month'] = month
    if not year:
        flash("Year is required to generate chart.", "warning")
        return redirect(url_for('home'))
    user_id = session.get('user_id')
    expenses = load_expenses(year=year, month=month, user_id=user_id)
    if not expenses:
        flash("No data found for the selected filters.", "warning")
        return redirect(url_for('home'))
    df = pd.DataFrame(expenses)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    plt.figure(figsize=(8, 6))
    if chart_type == 'pie':
        summary = df.groupby('category')['amount'].sum()
        values = summary.values.astype(np.float64)
        labels = [str(x) for x in summary.index.values]
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        title = f'Expense Summary by Category for {year}' + (f' Month {month}' if month else '')
        plt.title(title)
    elif chart_type == 'line':
        df['month'] = df['date'].dt.month
        monthly_sum = df.groupby('month')['amount'].sum().reindex(range(1, 13), fill_value=0)
        x_values = np.arange(1, 13, dtype=np.int32)
        y_values = monthly_sum.values.astype(np.float64)
        plt.plot(x_values, y_values, marker='o')
        plt.xticks(x_values)
        plt.xlabel('Month')
        plt.ylabel('Amount')
        title = f'Monthly Spending Trend for {year}'
        plt.title(title)
    elif chart_type == 'bar':
        summary = df.groupby('category')['amount'].sum()
        summary = pd.Series(summary).sort_values(ascending=False)
        x_values = np.arange(len(summary), dtype=np.int32)
        y_values = summary.values.astype(np.float64)
        labels = [str(x) for x in summary.index.values]
        plt.bar(x_values, y_values, color='skyblue')
        plt.xticks(x_values, labels, rotation=45, ha='right')
        plt.ylabel('Amount')
        title = f'Category-wise Spending for {year}' + (f' Month {month}' if month else '')
        plt.title(title)
    elif chart_type == 'budget_vs_actual':
        budget_data = get_yearly_budget_dict(year)
        df['month'] = df['date'].dt.month
        actual = df.groupby('month')['amount'].sum().reindex(range(1, 13), fill_value=0)
        months = np.arange(1, 13, dtype=np.int32)
        actual_values = actual.values.astype(np.float64)
        budget_values = np.array([float(budget_data.get(m, 0)) for m in range(1, 13)], dtype=np.float64)
        plt.plot(months, actual_values, label='Actual Spending', marker='o')
        plt.plot(months, budget_values, label='Budget', marker='o')
        plt.xticks(months)
        plt.xlabel('Month')
        plt.ylabel('Amount')
        title = f'Budget vs Actual Spending for {year}'
        plt.title(title)
        plt.legend()
    else:
        flash("Invalid chart type selected.", "warning")
        return redirect(url_for('home'))
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return render_template('summary_chart.html', chart_data=img_base64)

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    if request.method == 'GET':
        return render_template('chatbot.html')
        
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    user_input = data.get('message', '').strip()

    if not user_input:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        # Create a structured prompt for the AI
        prompt = f"""
        You are an AI assistant for an expense tracker app. Analyze the following user input and respond appropriately:
        
        User Input: {user_input}
        
        Please provide a response that:
        1. Understands the user's intent
        2. Provides helpful information about expense tracking
        3. Suggests relevant actions if needed
        
        Keep the response concise and focused on expense tracking.
        """

        response = gemini_ask(prompt)
        
        # Process the response and determine the appropriate action
        if any(keyword in user_input.lower() for keyword in ['add', 'new expense', 'spent']):
            return jsonify({
                "message": "I can help you add a new expense. Please use the 'Add Expense' form or provide the details in the format: 'Add expense of [amount] for [description] on [date]'"
            })
        elif any(keyword in user_input.lower() for keyword in ['show', 'view', 'list', 'expenses']):
            return jsonify({
                "message": "I can help you view your expenses. You can use the filters on the home page to view expenses by date, category, or search for specific expenses."
            })
        elif any(keyword in user_input.lower() for keyword in ['budget', 'set budget']):
            return jsonify({
                "message": "I can help you set or view your budget. Please use the 'Set Budget' page or provide the amount you want to set as your budget."
            })
        else:
            return jsonify({"message": response})

    except Exception as e:
        print(f"Error in chatbot: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

@app.route('/ask_gemini', methods=['POST'])
@login_required
def ask_gemini():
    try:
        data = request.get_json()
        prompt = data.get('message', '').strip()
        
        if not prompt:
            return jsonify({'reply': "Please send a message to ask."})

        # Add context to the prompt
        enhanced_prompt = f"""
        You are an AI assistant for an expense tracker app. The user asks:
        {prompt}
        
        Please provide a helpful response focused on expense tracking and financial management.
        """

        response = gemini_ask(enhanced_prompt)
        return jsonify({'reply': response})
        
    except Exception as e:
        print(f"Error in ask_gemini: {str(e)}")
        return jsonify({'reply': "I apologize, but I'm having trouble processing your request right now."})
    
if __name__ == '__main__':
    app.run(debug=True)
