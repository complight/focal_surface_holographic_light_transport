from odak.learn.wave.models import focal_surface_light_propagation


def make_model( settings):
    model = focal_surface_light_propagation(device=settings["general"]["device"])
    return model