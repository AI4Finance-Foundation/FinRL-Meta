from setuptools import setup

# Read requirements.txt, ignore comments
# try:
#     REQUIRES = list()
#     f = open("requirements.txt", "rb")
#     for line in f.read().decode("utf-8").split("\n"):
#         line = line.strip()
#         if "#" in line:
#             line = line[: line.find("#")].strip()
#         if line:
#             REQUIRES.append(line)
# except FileNotFoundError:
#     print("'requirements.txt' not found!")
#     REQUIRES = list()

setup(
    name="finrl_meta",
    version="0.3.0",
    author="Xiaoyang Liu, Jingyang Rui, Hongyang Yang",
    author_email="hy2500@columbia.edu",
    url="https://github.com/AI4Finance-Foundation/FinRL-Meta",
    license="MIT",
    # dependency_links=['git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2'],
    # install_requires=REQUIRES,
    description="FinRL­-Meta: A Universe of Near Real­ Market En­vironments for Data­-Driven Financial Reinforcement Learning",
    long_description="FinRL­-Meta: A Universe of Near Real­ Market En­vironments for Data­-Driven Financial Reinforcement Learning",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcment Learning",
    platform=["any"],
    python_requires=">=3.6",
)
