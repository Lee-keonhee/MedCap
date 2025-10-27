import yaml
import os


def load_config(config_path):
    """
    YAML config 파일 로드

    Args:
        config_path: config 파일 경로

    Returns:
        dict: config 내용
    """
    # TODO: 파일 존재 확인
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # TODO: YAML 로드

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

#
# def convert_paths(config):
#     """
#     Config 내 상대 경로를 절대 경로로 변환
#     """
#     # TODO: 구현


if __name__ == '__main__':
    # 테스트
    config = load_config('../../configs/base_config.yaml')

    print("=== Experiment ===")
    print(config['experiment'])

    print("\n=== Model ===")
    print(config['model'])

    print("\n=== Training ===")
    print(config['training'])