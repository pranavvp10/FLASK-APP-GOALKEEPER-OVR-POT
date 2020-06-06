from  pvp import  db,login_manager
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
class User(db.Model):
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer,primary_key=True, autoincrement=True)
    username = db.Column(db.String(20),unique=True, nullable=False)
    password = db.Column(db.String(20), unique=True, nullable=False)
    attr = db.relationship('Attr',lazy='select',backref=db.backref('user',lazy='joined'))
    res=db.relationship('Res',lazy='select',backref=db.backref('user',lazy='joined'))
    def __repr__(self):
        return  f"[{self.username} ,{self.password}]"
class Attr(db.Model):
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer,primary_key=True, autoincrement=True)
    age = db.Column(db.Integer)
    gd = db.Column(db.Integer)
    gh = db.Column(db.Integer)
    gk = db.Column(db.Integer)
    gp = db.Column(db.Integer)
    gr = db.Column(db.Integer)
    userid=db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    def __repr__(self):
        return f"[{self.age},{self.gd},{self.gh},{self.gk},{self.gp},{self.gr}]"
class Res(db.Model):
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    resl = db.Column(db.Integer)
    resid=db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    def __repr__(self):
        return f"[{self.resl}]"
