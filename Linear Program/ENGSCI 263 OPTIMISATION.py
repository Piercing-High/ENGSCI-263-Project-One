from pulp import *
import pandas as pd
import math
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUTE_POOLS_XLSX = PROJECT_ROOT / "Routes" / "route_pools.xlsx"


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def parse_stores(route_str):
    """
    Extracts the stores from a complete route.

    Example:
        "Warehouse -> Botany -> Manukau -> Warehouse"

    becomes:
        ["Botany", "Manukau"]

    The Warehouse is removed because it is only the starting/ending location
    and is not a store that needs to be covered.
    """

    return route_str.split(" -> ")[1:-1]


def linfox_cost(total_hours):
    """
    Calculates the cost of using Linfox for a route.

    Linfox charges $1400 for every 2-hour block.

    math.ceil() is used because even part of a 2-hour block requires
    payment for the entire block.

    Examples:
        1.5 hours -> 1 block -> $1400
        2.0 hours -> 1 block -> $1400
        2.1 hours -> 2 blocks -> $2800
        4.1 hours -> 3 blocks -> $4200
    """

    blocks = math.ceil(total_hours / 2)

    return blocks * 1400


def shed_penalty(store_name):
    """
    Returns the one-off cost of skipping (shedding) a store's delivery
    for the day, as described in the fuel-reduction proposal.

    Pak 'n Save stores cost more to skip ($1500) than other store
    types ($800), because store type is identifiable directly from
    the store's name prefix.

    Examples:
        "Pak 'n Save Manukau"   -> $1500
        "New World Remuera"     -> $800
        "Four Square Britomart" -> $800
    """

    if store_name.startswith("Pak 'n Save"):
        return 1500
    else:
        return 800


def build_route_vars(routes_df):
    """
    Creates one binary decision variable for every possible route.

    For each route r:

        x[r] = 1 -> the route is selected
        x[r] = 0 -> the route is not selected

    These variables are what the optimisation solver changes to find
    the cheapest combination of routes.
    """

    # Get the index/ID of every route in the DataFrame.
    route_ids = routes_df.index.tolist()

    # Create one integer variable between 0 and 1 for every route.
    #
    # Because the variable must be an integer, it can only be:
    #
    #       0 or 1
    #
    # This makes it a binary decision variable.

    return LpVariable.dicts(
        "Route",
        route_ids,
        0,
        1,
        LpInteger
    )


# =============================================================================
# 2. MAIN OPTIMISATION FUNCTION
# =============================================================================

def solve_routing(routes_df, day_name, truck_cap=40,
                   allow_shedding=False, shed_fraction=0.20):
    """
    Builds and solves the routing optimisation problem for one day.

    Parameters
    ----------
    routes_df:
        DataFrame containing the available routes for that day.

    day_name:
        Name of the day, e.g. "Weekday" or "Saturday".
        This is used to name the optimisation problem and print results.

    truck_cap:
        Maximum number of Fleet truck shifts available.
        Default = 40.

    allow_shedding:
        If False (default), every store MUST be delivered to - this is
        the baseline least-cost routing model.

        If True, up to `shed_fraction` of stores may instead be skipped
        for a one-off penalty cost, modelling the fuel-reduction
        proposal. The solver only sheds a store if doing so is cheaper
        overall than delivering to it.

    shed_fraction:
        Maximum proportion of stores that can be shed in a day.
        Default = 0.20 (20%), as specified in the fuel-reduction
        proposal. Only used when allow_shedding=True.

    Returns
    -------
    prob:
        The solved PuLP optimisation problem.

    routes_df:
        The final DataFrame containing Fleet and Linfox routes.

    x:
        The route decision variables.

    selected:
        List of routes selected by the optimiser.

    shed:
        Dictionary of shedding decision variables, keyed by store name.
        Empty dictionary if allow_shedding=False.
    """


    # -------------------------------------------------------------------------
    # 2.1 Make a copy
    # -------------------------------------------------------------------------
    #
    # We make a copy so that the original DataFrame loaded from Excel isn't
    # accidentally modified.
    #
    # This is particularly useful because the same function is used for both
    # weekday and Saturday, and for both the baseline and shedding versions.

    routes_df = routes_df.copy()


    # -------------------------------------------------------------------------
    # 2.2 Extract the stores from each route
    # -------------------------------------------------------------------------
    #
    # The original Excel file contains a route string.
    #
    # We create a "stores" column containing a list of the stores visited
    # by each route.

    routes_df["stores"] = routes_df["route"].apply(parse_stores)


    # -------------------------------------------------------------------------
    # 2.3 Create Fleet and Linfox versions of every route
    # -------------------------------------------------------------------------
    #
    # The original route pool represents routes that can be completed using
    # our own Fleet.
    #
    # Therefore, initially every route is labelled:
    #
    #       source = "Fleet"
    #
    # We then create a copy of every route and label that copy as:
    #
    #       source = "Linfox"
    #
    # This means that for every possible route, the optimisation can potentially
    # choose either:
    #
    #       Fleet version
    #
    # or:
    #
    #       Linfox version
    #
    # The Linfox version gets a cost calculated from the total route hours.

    routes_df["source"] = "Fleet"


    # Make a copy of all routes to create the Linfox alternatives.

    linfox_routes = routes_df.copy()

    # Mark these routes as outsourced to Linfox.

    linfox_routes["source"] = "Linfox"

    # Calculate the Linfox cost for each route.

    linfox_routes["cost"] = (
        linfox_routes["total_hours"].apply(linfox_cost)
    )


    # Combine the Fleet and Linfox routes into one route pool.
    #
    # For example, if there were originally 100 routes:
    #
    #       100 Fleet routes
    #       100 Linfox routes
    #
    # giving:
    #
    #       200 possible route choices.

    routes_df = pd.concat(
        [routes_df, linfox_routes],
        ignore_index=True
    )


    # -------------------------------------------------------------------------
    # 2.4 Create the binary route decision variables
    # -------------------------------------------------------------------------
    #
    # x[r] represents whether route r is selected.

    x = build_route_vars(routes_df)


    # -------------------------------------------------------------------------
    # 2.5 Create the optimisation problem
    # -------------------------------------------------------------------------
    #
    # We want to MINIMISE the total cost of the selected routes
    # (plus shedding penalties, if shedding is enabled).

    prob = LpProblem(
        f"{day_name}_Truck_Routing",
        LpMinimize
    )


    # =========================================================================
    # 3. COVERAGE CONSTRAINTS
    # =========================================================================
    #
    # First, find every unique store that appears in the route pool.
    #
    # For example:
    #
    #       ["Albany", "Botany", "Manukau", "Takapuna"]
    #
    # We need to create one coverage constraint for every store.

    all_stores = sorted(
        set(
            store
            for stores in routes_df["stores"]
            for store in stores
        )
    )


    # -------------------------------------------------------------------------
    # 3.0 Create shedding decision variables (only if enabled)
    # -------------------------------------------------------------------------
    #
    # shed[store] = 1 -> the store's delivery is skipped for the day
    # shed[store] = 0 -> the store's delivery goes ahead as normal
    #
    # These variables only get created if allow_shedding=True. If shedding
    # is not enabled, `shed` stays an empty dictionary and every coverage
    # constraint below falls back to the original baseline behaviour.

    if allow_shedding:
        shed = LpVariable.dicts("Shed", all_stores, 0, 1, LpInteger)
    else:
        shed = {}


    # -------------------------------------------------------------------------
    # 3.1 Require every store to be covered exactly once (or shed)
    # -------------------------------------------------------------------------
    #
    # We go through each store individually.
    #
    # For a particular store, we find all routes that visit that store.

    for store in all_stores:

        covering_routes = [
            r
            for r in routes_df.index
            if store in routes_df.loc[r, "stores"]
        ]


        # Baseline behaviour (allow_shedding=False):
        #
        #       x[2] + x[7] + x[15] = 1
        #
        # Exactly one selected route must visit this store - it cannot
        # be missed, and cannot be double-covered.
        #
        # Shedding behaviour (allow_shedding=True):
        #
        #       x[2] + x[7] + x[15] + shed[store] = 1
        #
        # Now EITHER a selected route visits the store (shed[store]
        # forced to 0, no penalty), OR the store is shed (shed[store]
        # forced to 1, penalty applies in the objective). Exactly one
        # of "delivered" or "shed" must be true - a store can still not
        # be double-covered, and cannot silently be missed without the
        # penalty being counted.

        if allow_shedding:
            prob += (
                lpSum(x[r] for r in covering_routes) + shed[store] == 1,
                f"Cover_{store}"
            )
        else:
            prob += (
                lpSum(x[r] for r in covering_routes) == 1,
                f"Cover_{store}"
            )


    # -------------------------------------------------------------------------
    # 3.2 Cap the number of stores that can be shed (only if enabled)
    # -------------------------------------------------------------------------
    #
    # The fuel-reduction proposal allows at most shed_fraction (20% by
    # default) of Auckland stores to have their delivery skipped on a
    # given day.

    if allow_shedding:
        max_sheds = shed_fraction * len(all_stores)

        prob += (
            lpSum(shed[store] for store in all_stores) <= max_sheds,
            "Max_Shed_Stores"
        )


    # =========================================================================
    # 4. FLEET TRUCK-SHIFT CONSTRAINT
    # =========================================================================
    #
    # We only count routes where source == "Fleet".
    #
    # Linfox routes do not use our own trucks, so they do not count towards
    # the Fleet truck capacity.
    #
    # For example, if:
    #
    #       20 trucks
    #       x 2 shifts per truck
    #
    # then:
    #
    #       truck_cap = 40
    #
    # meaning at most 40 Fleet routes can be selected.

    fleet_ids = routes_df[
        routes_df["source"] == "Fleet"
    ].index


    prob += (
        lpSum(x[r] for r in fleet_ids) <= truck_cap,
        "Max_Truck_Shifts"
    )


    # =========================================================================
    # 5. OBJECTIVE FUNCTION
    # =========================================================================
    #
    # We want to minimise total cost.
    #
    # For every route:
    #
    #       x[r] * route_cost
    #
    # If x[r] = 0:
    #
    #       the route contributes $0.
    #
    # If x[r] = 1:
    #
    #       the route's full cost is included.
    #
    # If shedding is enabled, we also add a penalty term for every store
    # that gets shed:
    #
    #       shed[store] * shed_penalty(store)
    #
    # If shed[store] = 0 (store delivered), this contributes $0.
    # If shed[store] = 1 (store skipped), the penalty cost is included.
    #
    # The solver therefore finds the cheapest combination of Fleet routes,
    # Linfox routes, and (if enabled) shed stores that satisfies all
    # constraints.

    route_cost_terms = [
        x[r] * routes_df.loc[r, "cost"]
        for r in routes_df.index
    ]

    if allow_shedding:
        shed_cost_terms = [
            shed[store] * shed_penalty(store)
            for store in all_stores
        ]
    else:
        shed_cost_terms = []

    prob += (
        lpSum(route_cost_terms + shed_cost_terms),
        "Total_Cost"
    )


    # =========================================================================
    # 6. SOLVE
    # =========================================================================

    prob.solve(PULP_CBC_CMD(msg=False))


    # =========================================================================
    # 7. IDENTIFY THE SELECTED ROUTES AND SHED STORES
    # =========================================================================
    #
    # After solving, x[r].varValue tells us the value chosen by the solver.
    #
    # We only want routes where:
    #
    #       x[r] = 1
    #
    # because these are the routes actually being used.

    selected = [
        r
        for r in routes_df.index
        if x[r].varValue == 1
    ]


    # Separate the selected routes into Fleet and Linfox routes.
    #
    # This allows us to see how much work is being done internally versus
    # outsourced.

    fleet_used = [
        r
        for r in selected
        if routes_df.loc[r, "source"] == "Fleet"
    ]

    linfox_used = [
        r
        for r in selected
        if routes_df.loc[r, "source"] == "Linfox"
    ]


    # Identify which stores (if any) were shed.

    if allow_shedding:
        shed_stores = [
            store
            for store in all_stores
            if shed[store].varValue == 1
        ]
    else:
        shed_stores = []


    # =========================================================================
    # 8. PRINT RESULTS
    # =========================================================================

    print(f"--- {day_name} ({'Shedding allowed' if allow_shedding else 'Baseline, no shedding'}) ---")

    print(
        "Status:",
        LpStatus[prob.status]
    )

    print(
        "Total Cost: $",
        value(prob.objective)
    )

    print(
        "Fleet routes used:",
        len(fleet_used)
    )

    print(
        "Linfox routes used:",
        len(linfox_used)
    )

    print(
        "Stores shed:",
        len(shed_stores), "/", len(all_stores)
    )

    if shed_stores:
        for store in shed_stores:
            print("   -", store, f"(penalty ${shed_penalty(store)})")

    print()


    # Return the model, routes, variables, selected routes and shed stores
    # so that we can inspect them later if required.

    return prob, routes_df, x, selected, shed_stores


# =============================================================================
# 9. LOAD THE RAW DATA
# =============================================================================
#
# We load the Excel sheets only once.
#
# The optimisation function will then process whichever DataFrame we give it.

weekday_raw = pd.read_excel(ROUTE_POOLS_XLSX, sheet_name="Weekday Pool")
saturday_raw = pd.read_excel(ROUTE_POOLS_XLSX, sheet_name="Saturday Pool")


# =============================================================================
# 10. SOLVE THE BASELINE PROBLEMS (NO SHEDDING)
# =============================================================================
#
# This is the least-cost routing schedule required by the first part of
# the assignment - every store must be delivered to.

wk_prob, wk_routes, wk_x, wk_selected, _ = solve_routing(
    weekday_raw,
    "Weekday",
    truck_cap=40,
    allow_shedding=False
)

sat_prob, sat_routes, sat_x, sat_selected, _ = solve_routing(
    saturday_raw,
    "Saturday",
    truck_cap=40,
    allow_shedding=False
)


# =============================================================================
# 11. SOLVE THE FUEL-REDUCTION PROBLEMS (SHEDDING ALLOWED)
# =============================================================================
#
# This models the fuel-reduction proposal - up to 20% of stores may have
# their delivery skipped for a one-off penalty cost, if doing so reduces
# total cost overall.

wk_shed_prob, wk_shed_routes, wk_shed_x, wk_shed_selected, wk_shed_stores = solve_routing(
    weekday_raw,
    "Weekday",
    truck_cap=40,
    allow_shedding=True,
    shed_fraction=0.20
)

sat_shed_prob, sat_shed_routes, sat_shed_x, sat_shed_selected, sat_shed_stores = solve_routing(
    saturday_raw,
    "Saturday",
    truck_cap=40,
    allow_shedding=True,
    shed_fraction=0.20
)

# =============================================================================
# 12. SUMMARY TABLE + SAVE SELECTED ROUTES
# =============================================================================

def summarise(prob, routes_df, selected, shed_stores, label):
    """Collect one scenario's key results into a dict."""
    fleet = [r for r in selected if routes_df.loc[r, "source"] == "Fleet"]
    linfox = [r for r in selected if routes_df.loc[r, "source"] == "Linfox"]
    return {
        "Scenario": label,
        "Total cost ($)": round(value(prob.objective), 2),
        "Fleet routes": len(fleet),
        "Linfox routes": len(linfox),
        "Stores shed": len(shed_stores),
    }

summary = pd.DataFrame([
    summarise(wk_prob,      wk_routes,      wk_selected,      [],             "Weekday baseline"),
    summarise(sat_prob,     sat_routes,     sat_selected,     [],             "Saturday baseline"),
    summarise(wk_shed_prob, wk_shed_routes, wk_shed_selected, wk_shed_stores, "Weekday + shedding"),
    summarise(sat_shed_prob,sat_shed_routes,sat_shed_selected,sat_shed_stores,"Saturday + shedding"),
])

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(summary.to_string(index=False))


def selected_routes_df(routes_df, selected, day_label):
    """Build a tidy table of the routes actually chosen for one scenario."""
    rows = []
    for r in selected:
        rows.append({
            "day": day_label,
            "source": routes_df.loc[r, "source"],
            "route": routes_df.loc[r, "route"],
            "stores": ", ".join(routes_df.loc[r, "stores"]),
            "n_stores": len(routes_df.loc[r, "stores"]),
            "total_hours": routes_df.loc[r, "total_hours"],
            "cost": routes_df.loc[r, "cost"],
        })
    return pd.DataFrame(rows)

# Save selected routes for ALL scenarios
scenarios = [
    (wk_routes,       wk_selected,       "Weekday",  "Weekday Baseline"),
    (sat_routes,      sat_selected,      "Saturday", "Saturday Baseline"),
    (wk_shed_routes,  wk_shed_selected,  "Weekday",  "Weekday Shedding"),
    (sat_shed_routes, sat_shed_selected, "Saturday", "Saturday Shedding"),
]

out_path = PROJECT_ROOT / "Linear Program" / "selected_routes.xlsx"
with pd.ExcelWriter(out_path) as writer:
    for routes_df, selected, day_label, sheet in scenarios:
        selected_routes_df(routes_df, selected, day_label).to_excel(
            writer, sheet_name=sheet, index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)

print(f"\nSaved selected routes to: {out_path}")