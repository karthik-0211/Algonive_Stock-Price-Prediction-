import numpy as np

def multi_step_prediction(model, data, scaler, steps=7):
    preds = []
    batch = data[-60:].reshape(1, 60, 1)

    for _ in range(steps):
        pred = model.predict(batch)[0]
        preds.append(pred)

        batch = np.append(batch[:,1:,:], [[pred]], axis=1)

    preds = scaler.inverse_transform(preds)
    return preds.flatten().tolist()