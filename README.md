(a) Expected Exposure (EE)
•	The expected exposure is the projected amount of money you stand to lose if default happens.
•	For a bond, this is the value at risk at each time step, typically the principal + interest payments.
•	In a zero-coupon bond, it’s just the discounted face value at each future date.
(b) Recovery Rate (RR) and Loss Given Default (LGD)
•	Recovery Rate (RR) = % of value recovered if default happens.
•	Loss Given Default (LGD) = Exposure × (1 − Recovery Rate).
Example: Exposure = 100, RR = 40% → LGD = 60.
(c) Probability of Default (PD)
•	PD is the probability that a bond defaults in a given year.
•	Can be:
o	Historical (actual) — based on past data, or
o	Risk-neutral — implied from market prices and used for valuation.
•	We often use conditional default probabilities (hazard rates):
PDₜ = base PD × probability of survival from earlier years.
(d) Expected Loss (EL)
•	EL = LGD × PD.
This gives expected value of losses in each period.
(e) Present Value of Expected Loss (PV_EL)
•	Each EL is discounted by the risk-free rate:
PV_ELt=ELt×DFtPV\_EL_t = EL_t \times DF_tPV_ELt=ELt×DFt 
where DF = 1/(1 + r_f)^t.
(f) Credit Valuation Adjustment (CVA)
•	CVA = Sum of PV_ELs = Present value of expected credit losses.
•	Represents how much the credit risk reduces the fair value of a bond.
(g) Credit Spread
•	The extra yield over the risk-free rate that compensates investors for credit risk.
•	Approximation:
Spread≈PD×(1−RR)\text{Spread} ≈ PD × (1 − RR)Spread≈PD×(1−RR)
(f) Python code for some examples
import numpy as np
import pandas as pd

def compute_cva(
    face_value=100,
    years=5,
    risk_free_rate=0.03,
    annual_pd=0.0125,
    recovery_rate=0.40
):
    """
    Computes Credit Valuation Adjustment (CVA) and fair value of a zero-coupon corporate bond.
    """

    data = []
    pos = 1.0  # Probability of survival at start

    for t in range(1, years + 1):
        # Exposure discounted to current year (risk-free)
        exposure = face_value / ((1 + risk_free_rate) ** (years - t))

        # Recovery and Loss Given Default
        recovery = exposure * recovery_rate
        lgd = exposure - recovery

        # Conditional Probability of Default (POD_t)
        pod = annual_pd * pos

        # Update survival probability for next period
        pos_next = pos - pod

        # Expected Loss = LGD * POD
        expected_loss = lgd * pod

        # Discount Factor
        df = 1 / ((1 + risk_free_rate) ** t)

        # PV of Expected Loss
        pv_el = expected_loss * df

        data.append([t, exposure, recovery, lgd, pod * 100, pos_next * 100, expected_loss, df, pv_el])

        # Update POS
        pos = pos_next

    df = pd.DataFrame(
        data,
        columns=[
            "Year", "Exposure", "Recovery", "LGD", "POD (%)", "POS (%)",
            "Expected Loss", "Discount Factor", "PV of Expected Loss"
        ]
    )

    cva = df["PV of Expected Loss"].sum()

    # Default-free bond price
    default_free_price = face_value / ((1 + risk_free_rate) ** years)

    # Fair value of risky bond
    fair_value = default_free_price - cva

    # Yield to maturity for risky bond
    ytm = (face_value / fair_value) ** (1 / years) - 1
    credit_spread = (ytm - risk_free_rate) * 10000  # in basis points

    return df.round(4), cva, fair_value, ytm, credit_spread


# Example: replicate Exhibit 2 (Five-year zero-coupon bond)
table, cva, fair_value, ytm, spread = compute_cva()

print("=== Credit Valuation Table ===")
print(table)
print("\nCVA =", round(cva, 4))
print("Fair Value of Risky Bond =", round(fair_value, 4))
print("Yield to Maturity =", round(ytm * 100, 2), "%")
print("Credit Spread =", round(spread, 2), "bps")



Result 
 
Interpretation of result : 
 
Investors require about 0.77% extra yield per year to compensate for the 1.25% annual default probability and 40% recovery assumption.
This aligns with typical investment-grade corporate bonds, which usually have credit spreads of 60–120 bps.
