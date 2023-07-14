from flask import Blueprint


router = Blueprint('home', __name__)


@router.route('/')
def home():
    return 'Hello !'
