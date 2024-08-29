import logging

def get_logger(path, output_to_terminal):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handlers = []
    file_handler = logging.FileHandler(path, mode='w')
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    if output_to_terminal:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    for handler in handlers:
        logger.addHandler(handler)

    return logger

def write_log(logger, embedding_dim, batchsize, nepochs, lr, lr_after, step_epoch, max_grad_norm, numberofdata, world_size, train_test_rate):
    logger.info("Experiment Config:")
    logger.info(f"embedding_dim: {embedding_dim}")
    logger.info(f"batchsize: {batchsize}")
    logger.info(f"nepochs: {nepochs}")
    logger.info(f"lr: {lr}")
    logger.info(f"lr_after: {lr_after}")
    logger.info(f"step_epoch: {step_epoch}")
    logger.info(f"max_grad_norm: {max_grad_norm}")
    logger.info(f"numberofdata: {numberofdata}")
    logger.info(f"world_size: {world_size}")
    logger.info(f"train_test_rate: {train_test_rate}")
