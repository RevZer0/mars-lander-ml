import random
from typing import List, Tuple, Type, TypeVar
import pyglet
import math

WORLD_SCALE = 3
WORLD_RATIO = 2.333 * WORLD_SCALE

Vec = TypeVar('Vec', bound='Vector')
Pt = TypeVar('Pt', bound='Point')
Color = Tuple[int, int, int]


class Point:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def distance(self, point: Pt) -> float:
        return math.sqrt(
            (self.x - point.x) ** 2 + (self.y - point.y) ** 2
        )

    def __repr__(self) -> str:
        return "x: %0.2f, y: %0.2f" % (self.x, self.y)


class Vector:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def scalar(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def dot(self, vector: Vec) -> float:
        return (self.x * vector.x) + (self.y * vector.y)

    def add(self, vector: Vec) -> Vec:
        return Vector(self.x + vector.x, self.y + vector.y)

    def mul(self, vector: Vec) -> Vec:
        return Vector(self.x * vector.x, self.y * vector.y)

    def mul_scalar(self, scalar: float) -> Vec:
        return Vector(self.x * scalar, self.y * scalar)

    def projection(self, vector: Vec) -> Vec:
        return vector.mul_scalar(vector.dot(self) / abs(vector.scalar() ** 2))

    def invert(self) -> Vec:
        return self.mul_scalar(-1)

    def __repr__(self) -> str:
        return "x: %0.2f, y: %0.2f, scalar: %0.2f" % (self.x, self.y, self.scalar())


class World:
    def __init__(self, width: int, height: int, terrain: List[Point]) -> None:
        self.width, self.height, self.terrain = width, height, terrain


class SimpleWorld(World):
    def __init__(self):
        super().__init__(
            7000,
            3000,
            [
                Point(0, 100), Point(1000, 500), Point(1500, 1500),
                Point(3000, 1000), Point(4000, 150), Point(5500, 150),
                Point(6999, 800)
            ]
        )


class POD:
    def __init__(self, initial_position: Point, fuel: int):
        self.x, self.y = initial_position.x, initial_position.y
        self.fuel = fuel
        self.h_speed, self.v_speed, self.rotation, self.power = (0, 0, 0, 0)
        self.target_rotation, self.target_thrust = (0, 0)
        self.velocity: Vector = Vector(0, 0)

    def set_rotation(self, angle: float) -> None:
        self.target_rotation = angle

    def set_thrust(self, thrust: float) -> None:
        self.target_thrust = thrust

    def move(self):
        self.x += self.velocity.x
        self.y += self.velocity.y


class Simulator:
    G = 3.711

    def __init__(self, unit: POD, world: World) -> None:
        self.pod = unit
        self.world = world
        self.time_elapsed: int = 0
        self.trajectory: List[Point] = []
        self.stopped = False

    def nearest_vertexes(self, count: int) -> List[Point]:
        for k in range(len(self.world.terrain)):
            if self.world.terrain[k].x <= self.pod.x <= self.world.terrain[k + 1].x:
                return [self.world.terrain[k], self.world.terrain[k + 1]]

    def simulate(self, dt: float) -> None:
        if self.stopped:
            return
        # limit thrust to add 1 by sec
        # limit rotation by 15 deg every 1 sec
        if self.pod.power != self.pod.target_thrust:
            power = abs(self.pod.rotation - self.pod.target_rotation)
            power = power if power < 1 else 1
            self.pod.power += (power * (-1 if self.pod.power > self.pod.target_thrust else 1)) * dt
        if self.pod.rotation != self.pod.target_rotation:
            rotation = abs(self.pod.rotation - self.pod.target_rotation)
            rotation = rotation if rotation < math.radians(15) else math.radians(15)
            self.pod.rotation += rotation * dt * (-1 if self.pod.rotation > self.pod.target_rotation else 1)
        self.time_elapsed += dt
        pw = self.pod.power * self.time_elapsed
        self.pod.velocity = Vector(pw * math.sin(self.pod.rotation), pw * math.cos(self.pod.rotation))
        gravity = Vector(0, -(Simulator.G * self.time_elapsed))
        self.pod.velocity = self.pod.velocity.add(gravity)
        self.trajectory.append(Point(self.pod.x, self.pod.y))
        self.pod.move()


class Utils:
    @staticmethod
    def coord_gtl(point: Tuple[float, float], scale: float = WORLD_RATIO):
        return Utils.gtl(point[0], scale), Utils.gtl(point[1], scale)

    @staticmethod
    def gtl(measure: float, scale: float = WORLD_RATIO):
        return measure / scale

    @staticmethod
    def gtl_point(point: Point):
        return Point(Utils.gtl(point.x), Utils.gtl(point.y))

    @staticmethod
    def coords_to_vertices(coords: List[Point]) -> Tuple[float, ...]:
        vertices = tuple()
        for point in coords:
            vertices += (point.x, point.y)
        return vertices


def vertex_to_vector(pod: POD, vertices: List[Point]) -> List[Vector]:
    return list(
        map(
            lambda vertex: Vector(pod.x - vertex.x, pod.y - vertex.y),
            vertices
        )
    )


def draw_terrain(points: List[Point], color: Color = (255, 0, 0)) -> None:
    indexes = list()
    for i in range(1, len(points)):
        indexes.append(i - 1)
        indexes.append(i)
    vertices = tuple(map(Utils.gtl, Utils.coords_to_vertices(points)))
    pyglet.graphics.draw_indexed(
        len(points),
        pyglet.gl.GL_LINES,
        indexes,
        ('v2f', vertices),
        ('c3B', color * len(points))
    )


def draw_pod(pod: POD) -> None:
    pod_size = 80
    top = Point(
        pod.x + (pod_size * math.sin(pod.rotation)),
        pod.y + (pod_size * math.cos(pod.rotation)),
    )
    bl = Point(
        pod.x - ((pod_size // 2) * math.cos(pod.rotation)),
        pod.y + ((pod_size // 2) * math.sin(pod.rotation))
    )
    br = Point(
        pod.x + ((pod_size // 2) * math.cos(pod.rotation)),
        pod.y - ((pod_size // 2) * math.sin(pod.rotation))
    )
    triangle = Utils.coords_to_vertices([
        Utils.gtl_point(top),
        Utils.gtl_point(bl),
        Utils.gtl_point(br),
    ])
    pyglet.graphics.draw_indexed(
        3,
        pyglet.gl.GL_TRIANGLES,
        [0, 1, 2, 0],
        ('v2f', triangle),
        ('c3B', (0, 255, 0) * 3)
    )


def draw_vector(start: Point, vec: Vector, color: Color) -> None:
    end = Utils.gtl_point(Point(abs(start.x - vec.x), abs(start.y - vec.y)))
    start = Utils.gtl_point(start)
    pyglet.graphics.draw(
        2,
        pyglet.gl.GL_LINES,
        ('v2f', (start.x, start.y, end.x, end.y)),
        ('c3B', color * 2)
    )


world = SimpleWorld()
window = pyglet.window.Window()
window.width, window.height = Utils.gtl(world.width), Utils.gtl(world.height)

simulators = []
for i in range(100):
    pod = POD(Point(2500, 2700), 5500)
    pod.set_rotation(math.radians(random.choice(range(-90, 90))))
    print(pod.target_rotation)
    pod.set_thrust(random.choice(range(1, 4)))
    simulators.append(Simulator(pod, world))

@window.event
def on_draw():
    global world, simulators
    window.clear()
    draw_terrain(world.terrain)
    for simulator in simulators:
        draw_terrain(simulator.trajectory)
        pod = simulator.pod
        draw_pod(pod)
        vertices = simulator.nearest_vertexes(2)
        #for vector in vertex_to_vector(pod, vertices):
        #    draw_vector(Point(pod.x, pod.y), vector, (0, 255, 255))
        ground = Vector(vertices[0].x - vertices[1].x, vertices[0].y - vertices[1].y)
        projector = (vertex_to_vector(pod, vertices)[0].projection(ground)).invert()
        projector_head = Point(abs(vertices[0].x - projector.x), abs(vertices[0].y - projector.y))
        collision = Vector(pod.x - projector_head.x, pod.y - projector_head.y)
        #draw_vector(Point(pod.x, pod.y), collision, (0, 255, 255))
        if collision.scalar() <= pod.velocity.scalar():
            simulator.stopped = True


def run_simulation(dt):
    global simulators
    for s in simulators:
        s.simulate(dt)


pyglet.clock.schedule_interval(run_simulation, 1 / 144)

if __name__ == '__main__':
    pyglet.app.run()
