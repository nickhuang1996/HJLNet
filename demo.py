from code.scripts.trainer import Trainer
from code.config.default_config import _C


if __name__ == '__main__':
    trainer = Trainer(cfg=_C)
    trainer.train()
    trainer.test()
    trainer.save_models()
