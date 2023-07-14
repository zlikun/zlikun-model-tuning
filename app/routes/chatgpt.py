from flask import Blueprint


router = Blueprint('chatgpt', __name__)


@router.route('/tunning')
def tunning():
    return 'Tunning ChatGPT !'
