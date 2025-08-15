import numpy as np
import pytest
from lbmgeom.shapes.circle import Circle
from lbmgeom.shapes.ellipse import Ellipse
from lbmgeom.shapes.rectangle import Rectangle
from lbmgeom.shapes.rounded_rect import RoundedRect
from lbmgeom.shapes.cassini_oval import CassiniOval
from lbmgeom.shapes.ops import Union, Intersection, Difference

def test_circle_sdf():
    c = Circle(0, 0, 5)
    assert np.isclose(c.sdf(0, 0), -5)
    assert np.isclose(c.sdf(5, 0), 0)
    assert c.sdf(6, 0) > 0

def test_ellipse_sdf_axis_aligned():
    e = Ellipse(0, 0, 4, 2)
    assert e.sdf(0, 0) < 0
    assert np.isclose(e.sdf(4, 0), 0, atol=1e-6)

def test_rectangle_sdf_no_rotation():
    r = Rectangle(0, 0, 4, 2)
    assert r.sdf(0, 0) < 0
    assert np.isclose(r.sdf(2, 0), 0, atol=1e-6)
    assert r.sdf(3, 0) > 0

def test_rounded_rect_sdf():
    rr = RoundedRect(0, 0, 4, 4, 1)
    assert rr.sdf(0, 0) < 0
    # A point far outside
    assert rr.sdf(10, 0) > 0

def test_cassini_oval_sdf_one_loop_two_loop():
    co1 = CassiniOval(0, 0, a=5, c=3)
    co2 = CassiniOval(0, 0, a=3, c=5)
    assert co1.sdf(0, 0) < 0
    assert co2.sdf(0, 0) > 0

def test_boolean_ops():
    c = Circle(0, 0, 1)
    r = Rectangle(0, 0, 2, 2)
    u = Union(c, r)
    i = Intersection(c, r)
    d = Difference(r, c)
    assert u.sdf(0, 0) < 0
    assert i.sdf(0, 0) < 0
    assert d.sdf(1.5, 0) < 0
    assert d.sdf(0, 0) > 0
