def train(max_epochs, model, optimizer, criterion, train_loader, test_loader, patience=10, min_delta=1e-8):
    
    best_test_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    
    for epoch in range(1, max_epochs + 1):
        train_loss, test_loss = 0.0, 0.0
        
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            model.backward(X_batch, criterion.backward(predictions, y_batch))
            optimizer.step()
            train_loss += loss.item() * X_batch.shape[0]
        
        model.eval()
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            test_loss += loss.item() * X_batch.shape[0]
        
        train_loss = train_loss / train_loader.num_samples()
        test_loss = test_loss / test_loader.num_samples()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Num epochs: {epoch}, best test loss: {best_test_loss}')
            break
    
    return train_losses, test_losses