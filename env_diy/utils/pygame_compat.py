from __future__ import annotations

try:
    import pygame as pygame
except ModuleNotFoundError:
    class Rect:
        def __init__(self, x: int, y: int, width: int, height: int):
            self.x = int(x)
            self.y = int(y)
            self.width = int(width)
            self.height = int(height)

        @property
        def left(self) -> int:
            return self.x

        @property
        def right(self) -> int:
            return self.x + self.width

        @property
        def top(self) -> int:
            return self.y

        @property
        def bottom(self) -> int:
            return self.y + self.height

        def colliderect(self, other: "Rect") -> bool:
            return not (
                self.right <= other.left
                or self.left >= other.right
                or self.bottom <= other.top
                or self.top >= other.bottom
            )

    class Surface:
        pass

    class _Draw:
        @staticmethod
        def rect(*args, **kwargs) -> None:
            return None

    class Vector2:
        def __init__(self, x: float | tuple[float, float] = 0.0, y: float = 0.0):
            if isinstance(x, tuple):
                self.x = float(x[0])
                self.y = float(x[1])
            else:
                self.x = float(x)
                self.y = float(y)

        def __sub__(self, other: "Vector2") -> "Vector2":
            return Vector2(self.x - other.x, self.y - other.y)

        def dot(self, other: "Vector2") -> float:
            return self.x * other.x + self.y * other.y

        def length(self) -> float:
            return (self.x**2 + self.y**2) ** 0.5

        def length_squared(self) -> float:
            return self.x**2 + self.y**2

        def normalize(self) -> "Vector2":
            result = Vector2(self.x, self.y)
            result.normalize_ip()
            return result

        def normalize_ip(self) -> None:
            current_length = self.length()
            if current_length <= 1e-12:
                return
            self.x /= current_length
            self.y /= current_length

    class _PygameCompat:
        K_UP = 273
        K_DOWN = 274
        K_LEFT = 276
        K_RIGHT = 275
        K_z = ord("z")
        K_x = ord("x")
        Rect = Rect
        Surface = Surface
        Vector2 = Vector2
        draw = _Draw()

    pygame = _PygameCompat()
