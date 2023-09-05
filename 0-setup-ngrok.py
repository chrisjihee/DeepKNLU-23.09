import logging

from flask import Flask

import flask_ngrok
from chrisbase.data import ProjectEnv, AppTyper
from chrisbase.io import LoggingFormat, run_command, hr
from flask_ngrok import run_with_ngrok

logger = logging.getLogger(__name__)
app = AppTyper()


def serve(ngrok_home, ngrok_vnum):
    app = Flask(__name__)
    run_with_ngrok(app, home=ngrok_home, ver=ngrok_vnum)  # Start ngrok when app is run

    @app.route("/")
    def hello():
        return "Hello World!"

    if __name__ == '__main__':
        app.run()
        # If address is in use, may need to terminate other sessions:
        # - Runtime > Manage Sessions > Terminate Other Sessions


@app.command()
def setup():
    env = ProjectEnv(project="DeepKNLU", msg_level=logging.INFO, msg_format=LoggingFormat.CHECK_36).info_args()
    ngrok_vnum = 3
    ngrok_home = env.working_path
    authtoken = "2NHZJsBLbOgcBLEmZTmuvJaOJM2_2VHnFpENJjTg5dwsSLjiE"

    logger.info(hr('='))
    ngrok_exe = flask_ngrok.install_ngrok(home=ngrok_home, ver=ngrok_vnum)
    logger.info(f"ngrok executable: {ngrok_exe}")
    ngrok_version = flask_ngrok.version_ngrok(home=ngrok_home, ver=ngrok_vnum)
    run_command("ls", "-al", ngrok_exe)
    logger.info(f"ngrok version: {ngrok_version}")

    logger.info(hr('='))
    ngrok_cfg = flask_ngrok.configure_ngrok(authtoken=authtoken, home=ngrok_home, ver=ngrok_vnum)
    run_command("ls", "-al", ngrok_cfg)
    run_command("cat", ngrok_cfg)

    logger.info(hr('='))
    serve(ngrok_home, ngrok_vnum)


if __name__ == "__main__":
    app()
