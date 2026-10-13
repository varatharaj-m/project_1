def register_blueprints(app):
    from . import audio_analysis, dashboard, health, history, image_analysis, news_analysis, reports, text_analysis, video_analysis
    from .auth import bp as auth_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(text_analysis.bp)
    app.register_blueprint(news_analysis.bp)
    app.register_blueprint(image_analysis.bp)
    app.register_blueprint(video_analysis.bp)
    app.register_blueprint(audio_analysis.bp)
    app.register_blueprint(history.bp)
    app.register_blueprint(reports.bp)
    app.register_blueprint(health.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(dashboard.api)
