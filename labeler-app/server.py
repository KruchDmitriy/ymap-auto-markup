import sys
from flask import *
from flask_login import *
from os import urandom
from wtforms import Form, StringField, PasswordField, validators
from optparse import OptionParser

from user_storage import UserStorage
from data_storage import TaskManager


app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)

users = UserStorage()
task_manager = TaskManager()


@app.route('/')
def index():
    if not current_user.is_anonymous:
        return redirect(url_for('map'))
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
        password = str(form.password.data)

        if not users.register(username, password):
            flash('User with this "username" already registered', 'danger')
            return render_template('register.html', form=form)
        login_user(users.find(username))
        flash('You are now registered and can log in', 'success')
        return redirect(url_for('map'))
    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_candidate = request.form['password']
        user = users.find(username)
        if not user.is_anonymous:
            if user.check_password(password_candidate):
                login_user(user, remember=True)
                return redirect(url_for('map'))
            else:
                error = 'Invalid password'
                return render_template('login.html', error=error)
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')


@login_manager.user_loader
def load_user(id):
    return users.find(id)


@app.route('/map/get_data', methods=['GET', 'POST'])
@login_required
def get_data():
    user = current_user.get_id()
    if not current_user.get_task():
        current_user.set_task(task_manager.next_task(user))
    return jsonify(task_manager.task_by_id(current_user.get_task(), user))


@app.route('/map/save_data', methods=['GET', 'POST'])
@login_required
def save_data():
    payload = request.get_json()

    if 'id' in payload:
        task_manager.append_result(current_user.get_id(), current_user.get_task(), payload)
    if 'complete' in payload:
        current_user.set_task(task_manager.next_task(current_user.get_id()))
    return ''


@app.route('/map')
@login_required
def map():
    return render_template('map.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@login_manager.unauthorized_handler
def handle_unauthorized():
    return redirect(url_for('login'))


@app.route('/finish', methods=['GET', 'POST'])
def finish():
    return render_template('finish.html')


# noinspection PyUnusedLocal
def enable_debug(*args, **kwargs):
    app.debug = True


parser = OptionParser()
parser.add_option("-d", "--debug",
                  action="callback", default=False, help="Enable flask server debugging",
                  callback=enable_debug)

if __name__ == "__main__":
    (options, args) = parser.parse_args(args=sys.argv[1:])
    app.secret_key = urandom(25)
    app.run()
