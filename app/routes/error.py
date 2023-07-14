from flask import Blueprint, render_template

router = Blueprint('errors', __name__)


@router.app_errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@router.app_errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500
