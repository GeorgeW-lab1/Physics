#!/usr/bin/env python3
"""
Australia macroeconomic calculator + exact budget-targeting via subset-sum decoder.

Keep subset_sum_paper.py in the same folder.

MONEY UNITS: AUD millions (integers for the solver).

Sources used for calibration:
- 2026-27 Commonwealth Budget: total expenses $833.262b; function expenses;
  receipts $798.1b; underlying cash deficit $31.542b; gross debt $1,051.0b;
  interest payments $43.364b. [Budget Paper No. 1 / Finance monthly statements]
- ABS nominal GDP: June-quarter 2026 GDP was $742.502b; annualised nominal GDP
  is approximately $2.970 trillion.
- Heritage Foundation 2026 Index: Australia overall score 80.1 and 12 pillar scores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from subset_sum_paper import solve, verify

ELASTICITY = 0.03   # illustrative: pp of annual real growth per freedom point above/below 60

FREEDOM_KEYS = [
    "property_rights", "judicial_effectiveness", "government_integrity",
    "tax_burden", "government_spending", "fiscal_health",
    "business_freedom", "labor_freedom", "monetary_freedom",
    "trade_freedom", "investment_freedom", "financial_freedom",
]


@dataclass
class SocialClass:
    name: str
    pop_share: float
    income_share: float
    tax_rate: float
    transfer_per_capita: float = 0.0   # AUD per person per year


@dataclass
class Country:
    name: str
    population: int
    gdp: float                          # AUD millions
    classes: List[SocialClass]
    spending: Dict[str, float]          # AUD millions
    debt: float                         # AUD millions
    interest_rate: float
    base_growth: float                  # real %/yr
    inflation: float                    # %/yr
    freedom: Dict[str, float] = field(default_factory=dict)

    def gdp_per_capita(self):
        return self.gdp * 1e6 / self.population

    def class_table(self):
        rows = []
        for c in self.classes:
            pop = c.pop_share * self.population
            inc = c.income_share * self.gdp
            rows.append({
                "class": c.name,
                "population": round(pop),
                "income_aud_m": round(inc, 1),
                "income_per_capita": round(inc * 1e6 / pop),
                "tax_aud_m": round(inc * c.tax_rate, 1),
                "transfers_aud_m": round(c.transfer_per_capita * pop / 1e6, 1),
            })
        return rows

    def gini(self):
        cs = sorted(self.classes, key=lambda c: c.income_share / c.pop_share)
        xp = yp = area = 0.0
        for c in cs:
            x, y = xp + c.pop_share, yp + c.income_share
            area += (x - xp) * (y + yp) / 2
            xp, yp = x, y
        return 1 - 2 * area

    def revenue(self):
        return sum(c.income_share * self.gdp * c.tax_rate for c in self.classes)

    def transfers(self):
        return sum(
            c.transfer_per_capita * c.pop_share * self.population / 1e6
            for c in self.classes
        )

    def interest(self):
        return self.debt * self.interest_rate

    def total_spending(self):
        return sum(self.spending.values()) + self.transfers() + self.interest()

    def deficit(self):
        return self.total_spending() - self.revenue()

    def fiscal_summary(self):
        d = self.deficit()
        return {
            "revenue_aud_m": round(self.revenue(), 1),
            "spending_aud_m": round(self.total_spending(), 1),
            "deficit(+)/surplus(-)_aud_m": round(d, 1),
            "deficit_%gdp": round(100 * d / self.gdp, 2),
            "debt_%gdp": round(100 * self.debt / self.gdp, 1),
            "interest_%revenue": round(100 * self.interest() / self.revenue(), 1),
            "tax_%gdp": round(100 * self.revenue() / self.gdp, 1),
        }

    def freedom_score(self):
        vals = [self.freedom[k] for k in FREEDOM_KEYS if k in self.freedom]
        return sum(vals) / len(vals) if vals else None

    def freedom_tier(self):
        s = self.freedom_score()
        if s is None:
            return "n/a"
        return (
            "Free" if s >= 80 else
            "Mostly Free" if s >= 70 else
            "Moderately Free" if s >= 60 else
            "Mostly Unfree" if s >= 50 else
            "Repressed"
        )

    def project(self, years=10):
        s = self.freedom_score() or 60
        g = (self.base_growth + ELASTICITY * (s - 60)) / 100
        gdp, debt, out = self.gdp, self.debt, []

        for year in range(1, years + 1):
            gdp *= 1 + g + self.inflation / 100
            debt += self.deficit()   # nominal deficit held constant
            out.append({
                "year": year,
                "gdp_aud_b": round(gdp / 1000, 1),
                "debt_aud_b": round(debt / 1000, 1),
                "debt_%gdp": round(100 * debt / gdp, 1),
            })
        return out

    def budget_fit(self, target: float, items: Optional[Dict[str, float]] = None,
                   seed=0, **kw):
        """Find spending line items whose rounded AUD-million amounts sum exactly to target."""
        items = items or self.spending
        names = list(items)
        amounts = [max(1, int(round(items[k]))) for k in names]
        target_int = int(round(target))

        solution, info = solve(amounts, target_int, seed=seed, **kw)
        if solution is None:
            return None, info

        chosen = [name for name, selected in zip(names, solution) if selected]
        assert verify(amounts, target_int, solution)
        return chosen, info


def australia():
    population = 27_000_000
    gdp = 2_970_000  # AUD millions; annualised from ABS June-quarter 2026 GDP

    # Illustrative household distribution; tax rates calibrated to roughly
    # reproduce the Budget's $798.1b receipts.
    classes = [
        SocialClass("Lower 40%", 0.40, 0.15, 0.090, 22_222),
        SocialClass("Middle 45%", 0.45, 0.48, 0.245, 4_938),
        SocialClass("Upper 15%", 0.15, 0.37, 0.370, 2_156),
    ]

    # 2026-27 Commonwealth expenses by function, excluding social security/welfare
    # and interest, which the model treats separately to prevent double-counting.
    spending = {
        "General public services": 32_499,
        "Defence": 52_055,
        "Public order and safety": 9_709,
        "Education": 57_349,
        "Health": 136_918,
        "Housing and community amenities": 11_454,
        "Recreation and culture": 6_292,
        "Fuel and energy": 15_207,
        "Agriculture, forestry and fishing": 4_748,
        "Mining, manufacturing and construction": 5_563,
        "Transport and communication": 15_338,
        "Other economic affairs": 13_773,
        "General purpose intergovernmental transactions": 111_062,
        "Natural disaster relief": 865,
        "Contingency reserve": 3_925,
        "Other purposes / residual": 4_404,
    }

    freedom = {
        "property_rights": 88.2,
        "judicial_effectiveness": 96.3,
        "government_integrity": 88.1,
        "tax_burden": 61.8,
        "government_spending": 56.6,
        "fiscal_health": 89.0,
        "business_freedom": 92.3,
        "labor_freedom": 61.5,
        "monetary_freedom": 77.2,
        "trade_freedom": 89.8,
        "investment_freedom": 80.0,
        "financial_freedom": 80.0,
    }

    return Country(
        name="Australia",
        population=population,
        gdp=gdp,
        classes=classes,
        spending=spending,
        debt=1_051_000,          # gross debt, AUD millions, 30 June 2027 estimate
        interest_rate=0.0413,    # implied by Budget interest payments / gross debt
        base_growth=1.75,        # 2026-27 real GDP growth forecast
        inflation=2.5,           # 2026-27 CPI forecast
        freedom=freedom,
    )


def demo():
    au = australia()

    print(f"== {au.name} ==")
    print(f"Population: {au.population:,}")
    print(f"Nominal GDP: A${au.gdp / 1e6:.2f} trillion")
    print(f"GDP per capita: A${au.gdp_per_capita():,.0f}")
    print(f"Gini (3-class approximation): {au.gini():.3f}")

    print("\nClasses:")
    for row in au.class_table():
        print("  ", row)

    print("\nFiscal summary:")
    for k, v in au.fiscal_summary().items():
        print(f"  {k}: {v}")

    score = au.freedom_score()
    print(f"\nEconomic freedom: {score:.1f} ({au.freedom_tier()})")

    print("\n10-year outlook (illustrative):")
    for row in au.project(10)[::3]:
        print("  ", row)

    target = 30_000  # A$30 billion
    print(f"\nBudget fit: which line items sum EXACTLY to A${target:,}m of cuts?")
    chosen, info = au.budget_fit(target)
    print("  ", chosen)
    print("  ", info["how"] if info else "")

    if chosen:
        total = sum(au.spending[k] for k in chosen)
        print(f"   Check: A${total:,}m")


if __name__ == "__main__":
    demo()
