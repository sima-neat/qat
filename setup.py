import setuptools
import subprocess
import os
import yaml


def get_version(pkg_dir, main_branch='master'):
    with open('VERSION.in') as fh:
        vinfo = yaml.load(fh.read(), Loader=yaml.BaseLoader)
        version = '.'.join([vinfo['major'], vinfo['minor'], vinfo['patch']])

    if 'GIT_HASH' in os.environ:
        # In tox the .git directory is not available so do this instead
        vinfo['git_hash'] = os.environ['GIT_HASH']
    else:
        proc = subprocess.Popen(
            ['git', 'log', '-1', '--format=%h'], stdout=subprocess.PIPE
        )
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError('ERROR: unable to execute git log')
        vinfo['git_hash'] = proc.stdout.read().decode('utf-8').rstrip()

    if not os.path.exists(pkg_dir):
        os.makedirs(pkg_dir)

    with open(pkg_dir + '/VERSION', 'w') as fh:
        print(vinfo)
        print(pkg_dir)
        fh.write(yaml.dump(vinfo))

    # This comes from Jenkins for upstream/downstream builds
    if 'DEV_VERSION' in os.environ:
        version = version + '.dev0+' + os.environ['DEV_VERSION']

    return version


used_deps = []


def get_package(env, pkg_name, pkg_version):
    if env + '_BRANCH' in os.environ and env + '_VERSION' in os.environ:
        branch = os.environ[env + '_BRANCH']
        version = os.environ[env + '_VERSION']

        if version == pkg_version or pkg_version == 'latest':
            deps = {}

            if os.path.exists('DEPS'):
                # Append dependency to DEPS file
                with open('DEPS', 'r') as fh:
                    deps = yaml.load(fh.read(), Loader=yaml.BaseLoader)

            deps[env] = '%s==%s.dev0+%s' % (pkg_name, version, branch)

            with open('DEPS', 'w') as fh:
                fh.write(yaml.dump(deps))

            return '%s==%s.dev0+%s' % (pkg_name, version, branch)

    if pkg_version == 'latest':
        return '%s' % (pkg_name)
    else:
        return '%s==%s' % (pkg_name, pkg_version)

docs_require = list()
#tests_require = list()

tests_require = [
    "pytest==7.0.0",
    "pytest-cov",
    "pytest-timeout",
    "pytest-randomly",
    "pytorch-lightning==2.4.0"
]

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text(encoding='utf-8')

with open('requirements.txt') as f:
    required = f.read().splitlines() # get the list of required libraries from requirements.txt

setuptools.setup(
    install_requires=required,
    extras_require={
        'docs': docs_require,
        'tests': tests_require,
        'dev': docs_require + tests_require,
    },
    python_requires='>=3.10',
    name='swml-qat',
    version=get_version('swml_qat'),
    author='SiMa.ai',
    author_email='support@sima.ai',
    description='SiMa.ai QAT implementation',
    long_description=long_description,
    url='https://sima.ai',
    packages=setuptools.find_packages(),
    package_data={'swml_qat': ['VERSION']},
    package_dir={'swml_qat': 'swml_qat'},
    license='Proprietary',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
)
