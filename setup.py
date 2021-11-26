from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name='src',
    description='Code for Adversarial Attacks in the Audio Domain: A Hands-On Introduction in PyTorch',
    version='0.0.1',
    author="Patrick O'Reilly",
    author_email='patrick.oreilly2024@u.northwestern.edu',
    url='https://github.com/oreillyp/adv_audio_intro',
    install_requires=[
        'torch',
        'torchaudio',
        'matplotlib',
        'numpy'
    ],
    packages=['src'],
    long_description=long_description,
    long_description_content_type='text.markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT'
)
