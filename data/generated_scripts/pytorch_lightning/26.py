def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return [optimizer], [scheduler]