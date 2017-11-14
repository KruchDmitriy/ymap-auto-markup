from flask import *
from os import urandom
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps

from user_storage import UserStorage
from data_storage import DataStorage, MarkupStorage
from user_tasks import TaskManager


app = Flask(__name__)

users = UserStorage()
data = DataStorage()
markup = MarkupStorage()
taskManager = TaskManager()


@app.route('/')
def index():
    return render_template('home.html')

class RegisterForm(Form):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [
            validators.DataRequired(),
            validators.EqualTo('confirm', message='Passwords do not match')
        ])
    confirm = PasswordField('Confirm Password')


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        if not users.register(username, password):
            flash('User with this "username" already registered', 'danger')
            return render_template('register.html', form=form)

        flash('You are now registered and can log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_candidate = request.form['password']

        if users.find(username):
            password = users.get_password(username)

            if sha256_crypt.verify(password_candidate, password):
                session['logged_in'] = True
                session['username'] = username

                return redirect(url_for('map'))
            else:
                error = 'Invalid password'
                return render_template('login.html', error=error)
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')

def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

@app.route('/map/get_data', methods=['GET', 'POST'])
@is_logged_in
def get_data():
    next_task = taskManager.next_task(session['username'])
    return jsonify(next_task)

@app.route('/map/save_data', methods=['GET', 'POST'])
@is_logged_in
def save_data():
    data = request.get_json()["data"]
    for obj in data:
        markup.append_json(session['username'], obj)
    markup.dump(session['username'])
    return ''

@app.route('/map')
@is_logged_in
def map():
    return render_template('map.html')

@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))

@app.route('/finish', methods=['GET', 'POST'])
def finish():
    return render_template('finish.html')


if __name__ == "__main__":
    app.secret_key = urandom(25)
    app.run()
