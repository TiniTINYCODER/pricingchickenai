def calculate_price(base_price, predicted_demand, remaining_stock):

    demand_ratio = predicted_demand / remaining_stock

    if demand_ratio > 1:
        price = base_price * 1.05

    elif demand_ratio < 0.5:
        price = base_price * 0.9

    else:
        price = base_price

    return round(price,2)