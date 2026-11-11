
import pygame
import networkx as nx
import random
import math
import sys

# --------------------------------------------------
# 1. INITIAL SETTINGS
# --------------------------------------------------

pygame.init()

WIDTH = 1200
HEIGHT = 780
PANEL_WIDTH = 300

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ACO - Wireless Sensor Network Simulation")

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 19)
small_font = pygame.font.SysFont("Arial", 14)
title_font = pygame.font.SysFont("Arial", 25, bold=True)

# Colours
BACKGROUND = (12, 18, 32)
PANEL = (22, 31, 49)
WHITE = (240, 245, 255)
GREEN = (40, 210, 120)
YELLOW = (245, 195, 55)
RED = (245, 75, 75)
GREY = (100, 110, 125)
CYAN = (40, 220, 240)
PURPLE = (180, 100, 255)
BLUE = (70, 145, 255)
ORANGE = (255, 165, 65)

# Network settings
ROWS = 5
COLS = 5

SOURCE = 10       # Middle-left node
SINK = 14         # Middle-right node

PACKET_INTERVAL = 0.8
TRANSMISSION_COST = 1.8

# ACO parameters
ALPHA = 1.2
BETA = 2.0
EVAPORATION = 0.96
REINFORCEMENT = 1.2

# --------------------------------------------------
# 2. CREATE THE NETWORK
# --------------------------------------------------

G = nx.Graph()

positions = {}

start_x = 90
start_y = 110
spacing_x = 145
spacing_y = 130

for row in range(ROWS):
    for col in range(COLS):

        node = row * COLS + col

        G.add_node(node)

        positions[node] = (
            start_x + col * spacing_x,
            start_y + row * spacing_y
        )

# Connect neighbouring nodes, including diagonal neighbours

for node in G.nodes():

    row = node // COLS
    col = node % COLS

    for other in G.nodes():

        other_row = other // COLS
        other_col = other % COLS

        row_difference = abs(row - other_row)
        col_difference = abs(col - other_col)

        if (
            node != other
            and row_difference <= 1
            and col_difference <= 1
        ):

            G.add_edge(node, other)

            G.edges[node, other]["pheromone"] = 1.0
            G.edges[node, other]["quality"] = random.uniform(0.7, 1.0)

# --------------------------------------------------
# 3. BATTERY LEVELS
# --------------------------------------------------

energy = {}

def reset_energy():

    global energy

    energy = {}

    for node in G.nodes():

        energy[node] = 100.0

# --------------------------------------------------
# 4. PACKET VARIABLES
# --------------------------------------------------

packets = []

delivered = 0
dropped = 0

packet_timer = 0
evaporation_timer = 0

paused = False
routing_mode = "ACO"

# --------------------------------------------------
# 5. RESET SIMULATION
# --------------------------------------------------

def reset_simulation():

    global packets
    global delivered, dropped
    global packet_timer, evaporation_timer

    packets = []

    delivered = 0
    dropped = 0

    packet_timer = 0
    evaporation_timer = 0

    reset_energy()

    for u, v in G.edges():

        G.edges[u, v]["pheromone"] = 1.0
        G.edges[u, v]["quality"] = random.uniform(0.7, 1.0)

# --------------------------------------------------
# 6. SELECT THE NEXT NODE
# --------------------------------------------------

def choose_next_node(current, visited):

    available = [
        node for node in G.neighbors(current)
        if node not in visited and energy[node] > 0
    ]

    if not available:
        return None

    # Shortest-path comparison mode

    if routing_mode == "Shortest-path":

        active_nodes = [
            node for node in G.nodes()
            if energy[node] > 0 or node in (SOURCE, SINK)
        ]

        active_graph = G.subgraph(active_nodes)

        try:

            path = nx.shortest_path(
                active_graph,
                source=current,
                target=SINK,
                weight="weight"
            )

            if len(path) > 1:
                return path[1]

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        return None

    # ACO routing mode

    scores = []

    for node in available:

        pheromone = G.edges[current, node]["pheromone"]

        quality = G.edges[current, node]["quality"]

        battery_factor = max(energy[node], 1) / 100

        distance = math.dist(
            positions[current],
            positions[node]
        )

        heuristic = (
            quality
            * battery_factor
            / distance
        )

        score = (
            (pheromone ** ALPHA)
            * (heuristic ** BETA)
        )

        scores.append(max(score, 0.000001))

    # Probabilistic next-hop selection

    total = sum(scores)

    probabilities = [
        score / total for score in scores
    ]

    return random.choices(
        available,
        weights=probabilities,
        k=1
    )[0]

# --------------------------------------------------
# 7. CREATE A FORWARD ANT (FANT)
# --------------------------------------------------

def create_fant():

    packets.append({

        "type": "FANT",

        "path": [SOURCE],

        "step": 0,

        "target": None,

        "progress": 0.0,

        "speed": 1.5

    })

# --------------------------------------------------
# 8. REINFORCE PHEROMONE
# --------------------------------------------------

def reinforce_path(path):

    if len(path) < 2:
        return

    path_length = len(path) - 1

    minimum_energy = min(
        energy[node] for node in path
    )

    energy_factor = minimum_energy / 100

    deposit = (
        REINFORCEMENT
        / path_length
        * energy_factor
    )

    for i in range(len(path) - 1):

        u = path[i]
        v = path[i + 1]

        G.edges[u, v]["pheromone"] += deposit

# --------------------------------------------------
# 9. UPDATE PACKETS
# --------------------------------------------------


def update_packets(dt):

    global delivered, dropped

    new_packets = []

    for packet in packets:

        kind = packet["type"]
        path = packet["path"]

        current = path[packet["step"]]

        # --------------------------------
        # FANT: Forward Ant
        # --------------------------------

        if kind == "FANT":

            # FANT has reached the sink
            if current == SINK:

                new_packets.append({

                    "type": "BANT",

                    "path": list(reversed(path)),

                    "step": 0,

                    "target": None,

                    "progress": 0.0,

                    "speed": 1.8

                })

                continue

            # Select next node
            if packet["target"] is None:

                next_node = choose_next_node(
                    current,
                    path
                )

                if next_node is None:

                    dropped += 1
                    continue

                packet["target"] = next_node

                # Reduce transmitting node battery
                if current not in (SOURCE, SINK):

                    energy[current] = max(
                        0,
                        energy[current] - TRANSMISSION_COST
                    )

        # --------------------------------
        # BANT: Backward Ant
        # --------------------------------

        else:

            # BANT has returned to source
            if packet["step"] >= len(path) - 1:

                reinforce_path(path)

                delivered += 1
                continue

            if packet["target"] is None:

                packet["target"] = path[
                    packet["step"] + 1
                ]

        # --------------------------------
        # MOVE THE PACKET
        # --------------------------------

        packet["progress"] += packet["speed"] * dt

        if packet["progress"] >= 1:

            arrived_node = packet["target"]

            # Record FANT's visited nodes
            if kind == "FANT":

                packet["path"].append(arrived_node)

            packet["step"] += 1

            packet["target"] = None

            packet["progress"] = 0.0

        new_packets.append(packet)

    # Update packet list after processing
    packets[:] = new_packets


          

# --------------------------------------------------
# 10. PHEROMONE EVAPORATION
# --------------------------------------------------

def evaporate_pheromones():

    for u, v in G.edges():

        value = G.edges[u, v]["pheromone"]

        G.edges[u, v]["pheromone"] = max(
            0.05,
            value * EVAPORATION
        )

# --------------------------------------------------
# 11. NODE COLOURS
# --------------------------------------------------

def get_node_colour(node):

    if node == SOURCE:
        return CYAN

    if node == SINK:
        return PURPLE

    battery = energy[node]

    if battery <= 0:
        return GREY

    elif battery < 20:
        return RED

    elif battery < 60:
        return YELLOW

    else:
        return GREEN

# --------------------------------------------------
# 12. DRAW NETWORK LINKS
# --------------------------------------------------

def draw_links():

    for u, v in G.edges():

        x1, y1 = positions[u]
        x2, y2 = positions[v]

        pheromone = G.edges[u, v]["pheromone"]

        brightness = min(
            255,
            int(65 + pheromone * 65)
        )

        colour = (
            35,
            brightness,
            min(255, brightness + 35)
        )

        thickness = min(
            7,
            max(1, int(pheromone * 2))
        )

        pygame.draw.line(
            screen,
            colour,
            (x1, y1),
            (x2, y2),
            thickness
        )

# --------------------------------------------------
# 13. DRAW SENSOR NODES
# --------------------------------------------------

def draw_nodes():

    for node in G.nodes():

        x, y = positions[node]

        colour = get_node_colour(node)

        # Outer glow

        pygame.draw.circle(
            screen,
            (35, 48, 68),
            (x, y),
            25
        )

        # Main node

        pygame.draw.circle(
            screen,
            colour,
            (x, y),
            17
        )

        pygame.draw.circle(
            screen,
            WHITE,
            (x, y),
            17,
            2
        )

        # Node number

        label = small_font.render(
            str(node),
            True,
            BACKGROUND
        )

        screen.blit(
            label,
            label.get_rect(center=(x, y))
        )

        # Battery percentage

        battery_label = small_font.render(
            f"{energy[node]:.0f}%",
            True,
            WHITE
        )

        screen.blit(
            battery_label,
            (x - 19, y + 24)
        )

# --------------------------------------------------
# 14. DRAW MOVING PACKETS
# --------------------------------------------------

def draw_packets():

    for packet in packets:

        path = packet["path"]

        current = path[packet["step"]]

        target = packet["target"]

        if target is None:
            x, y = positions[current]

        else:

            x1, y1 = positions[current]
            x2, y2 = positions[target]

            progress = packet["progress"]

            x = x1 + (x2 - x1) * progress
            y = y1 + (y2 - y1) * progress

        colour = (
            ORANGE if packet["type"] == "FANT"
            else BLUE
        )

        pygame.draw.circle(
            screen,
            colour,
            (int(x), int(y)),
            9
        )

        pygame.draw.circle(
            screen,
            WHITE,
            (int(x), int(y)),
            9,
            1
        )

# --------------------------------------------------
# 15. DRAW INFORMATION PANEL
# --------------------------------------------------

def draw_panel():

    pygame.draw.rect(
        screen,
        PANEL,
        (WIDTH - PANEL_WIDTH, 0, PANEL_WIDTH, HEIGHT)
    )

    x = WIDTH - PANEL_WIDTH + 20
    y = 25

    title = title_font.render(
        "ACO NETWORK",
        True,
        CYAN
    )

    screen.blit(title, (x, y))

    y += 55

    information = [

        f"Routing Mode: {routing_mode}",

        f"Packets Delivered: {delivered}",

        f"Packets Dropped: {dropped}",

        f"Active Packets: {len(packets)}",

        f"Live Nodes: {sum(1 for n in G.nodes() if energy[n] > 0)}",

        f"Average Battery: {sum(energy.values()) / len(energy):.1f}%"

    ]

    for item in information:

        text = font.render(
            item,
            True,
            WHITE
        )

        screen.blit(text, (x, y))

        y += 38

    y += 15

    screen.blit(
        font.render("NODE BATTERY LEVELS", True, YELLOW),
        (x, y)
    )

    y += 35

    for colour, label in [

        (GREEN, "Healthy: above 60%"),
        (YELLOW, "Moderate: 20-60%"),
        (RED, "Low: below 20%"),
        (GREY, "Depleted: 0%")

    ]:

        pygame.draw.circle(
            screen,
            colour,
            (x + 8, y + 9),
            7
        )

        screen.blit(
            small_font.render(label, True, WHITE),
            (x + 25, y)
        )

        y += 28

    y += 15

    screen.blit(
        font.render("PACKET LEGEND", True, YELLOW),
        (x, y)
    )

    y += 35

    pygame.draw.circle(screen, ORANGE, (x + 8, y + 8), 7)

    screen.blit(
        small_font.render("FANT: Forward Ant", True, WHITE),
        (x + 25, y)
    )

    y += 30

    pygame.draw.circle(screen, BLUE, (x + 8, y + 8), 7)

    screen.blit(
        small_font.render("BANT: Backward Ant", True, WHITE),
        (x + 25, y)
    )

    y += 55

    screen.blit(
        font.render("CONTROLS", True, CYAN),
        (x, y)
    )

    y += 35

    controls = [
        "SPACE - Pause / Resume",
        "A - Switch routing mode",
        "R - Reset simulation",
        "ESC - Exit"
    ]

    for item in controls:

        screen.blit(
            small_font.render(item, True, WHITE),
            (x, y)
        )

        y += 27

# --------------------------------------------------
# 16. MAIN SIMULATION LOOP
# --------------------------------------------------

reset_simulation()

running = True

while running:

    dt = min(clock.tick(60) / 1000, 0.05)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                running = False

            elif event.key == pygame.K_SPACE:

                paused = not paused

            elif event.key == pygame.K_r:

                reset_simulation()

            elif event.key == pygame.K_a:

                if routing_mode == "ACO":

                    routing_mode = "Shortest-path"

                else:

                    routing_mode = "ACO"

                packets.clear()

    if not paused:

        packet_timer += dt
        evaporation_timer += dt

        # Generate new FANT packets

        if packet_timer >= PACKET_INTERVAL:

            create_fant()

            packet_timer = 0

        # Update moving packets

        update_packets(dt)

        # Evaporate pheromone periodically

        if evaporation_timer >= 0.5:

            evaporate_pheromones()

            evaporation_timer = 0

    # ----------------------------------------------
    # DRAW EVERYTHING
    # ----------------------------------------------

    screen.fill(BACKGROUND)

    draw_links()

    draw_nodes()

    draw_packets()

    draw_panel()

    if paused:

        pause_text = title_font.render(
            "SIMULATION PAUSED",
            True,
            YELLOW
        )

        screen.blit(
            pause_text,
            (280, 20)
        )

    pygame.display.flip()

pygame.quit()
sys.exit()
