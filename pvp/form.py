from flask_wtf import FlaskForm
from  wtforms import  StringField,PasswordField,SubmitField,IntegerField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
from  wtforms.validators import data_required,length
class SigninForm(FlaskForm):
    username = StringField('Username',validators=[length(min=2,max=10)])
    password = PasswordField('Password',validators=[data_required()])
    submit = SubmitField('Sign In')
class attrform(FlaskForm):
    age = IntegerField('  Age ', validators=[data_required()])
    gd = IntegerField('  Gd  ', validators=[data_required()])
    gh = IntegerField('  Gh ', validators=[data_required()])
    gk = IntegerField('  Gk  ', validators=[data_required()])
    gp = IntegerField('  Gp  ', validators=[data_required()])
    gr = IntegerField('  Gr  ', validators=[data_required()])
    enter= SubmitField('Enter')
class signupform(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

