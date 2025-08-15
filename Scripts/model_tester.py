



def test_model(model, test, num_preds, batch_size):
    dataloader = torch.utils.data.DataLoader(list(test), batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            text_input, text_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            metadata, targets = batch['metadata'].to(device), batch['targets'].to(device)

            predictions = model(text_input, text_mask, metadata, num_preds)
            predictions = metrics.get_best_point(predictions.cpu().numpy())

            all_predictions.append(predictions)
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = metrics.get_results(all_predictions, all_targets)
    metrics.print_metrics(results)

    return results