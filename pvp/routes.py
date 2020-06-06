from flask import render_template,flash,redirect,url_for,request
from pvp import app,bcrypt
from pvp.form import signupform,SigninForm,attrform
from pvp.models import User,Res,Attr,db
from flask_login import current_user
from pvp.player_test import *
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')
@app.route("/about")
def about():
    return render_template('about.html', title='about')
@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = signupform()
    if form.validate_on_submit():
        db.create_all()
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('home'))
    return render_template('signup.html', title='Signup', form=form)
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    form = SigninForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            return  redirect(url_for('inp'))
        else:
            return "invalid username/password"
    return render_template('signin.html', titile='Signin',form=form)
@app.route('/inp', methods=['GET', 'POST'])
def inp():
    form = attrform()
    if form.validate_on_submit():
        flash('attrs entered')
        for i in range(1,(User.query.count()+1)):
            user = User.query.get(i)
        e=Attr(age=form.age.data,gd=form.gd.data,gh=form.gh.data,gk=form.gk.data,gp=form.gp.data,gr=form.gr.data,userid=user.id)
        db.session.add(e)
        db.session.commit()
        return  redirect(url_for('result'))
    return render_template('inp.html', form=form)
@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        form=attrform()
        new_data = OrderedDict([('age', form.age.data), ('gkdiving', form.gd.data), ('gkhandling', form.gh.data),
                                ('gkkicking', form.gk.data), ('gkpositioning', form.gp.data),
                                ('gkreflexes', form.gr.data)])
        new_data = pd.Series(new_data).values.reshape(1, -1)
        prediction = int((rf_final.predict(new_data)))
        for i in range(1,(User.query.count()+1)):
            user = User.query.get(i)
        y=Res(resl=prediction,resid=user.id)
        db.session.add(y)
        db.session.commit()
    return render_template('result.html',prediction=prediction)