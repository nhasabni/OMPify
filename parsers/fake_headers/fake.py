import os
import re
from functools import reduce


# REPOS_DIR = '/home/talkad/Downloads/thesis/data_gathering_script/git_repos'
REPOS_DIR = '/home/talkad/Downloads/thesis/data_gathering_script/asd'
REPOS_OMP_DIR = '/home/talkad/Downloads/thesis/data_gathering_script/repositories_openMP'
FAKE_DEFINES  = '_fake_defines.h'
FAKE_TYPEDEFS = '_fake_typedefs.h'
COMMON_FAKE_DEFINES  = '_common_fake_defines.h'
COMMON_FAKE_TYPEDEFS = '_common_fake_typedefs.h'
FAKE_INCLUDE  = f'#include \"{FAKE_DEFINES}\"\n#include \"{FAKE_TYPEDEFS}\"'


def get_headers(repo_dir, repo_name):
    '''
    For a given repo, return all headers file relative path 
    '''
    headers = []
    path_length = len(repo_dir) + len (repo_name) + 2              # remove two '/'
    omp_repo = os.path.join(repo_dir, repo_name)

    for idx, (root, dirs, files) in enumerate(os.walk(omp_repo)):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            
            if ext == '.h':
                headers.append(os.path.join(root, file_name)[path_length: ])

    return headers


def join_splited_lines(code_buf, delimiter='\\'):
    '''
    Several c files are splitting lines of code using \ token. For instance:
        #pragma omp parallel for private(i, test, x, y) \
            default(shared) schedule(dynamic)
    pycparser fail to process this files. So we update this lines to be a single line.
    '''
    code = []
    splitted_line = False

    for line in code_buf.split('\n'):
        if not splitted_line and len(line) > 0 and line[-1] == delimiter:
            code.append(line[:-1])
            splitted_line = True
        elif splitted_line and len(line) > 0 and line[-1] == delimiter:
            code[-1] += line[:-1]
        elif splitted_line:
            code[-1] += line
            splitted_line = False
        else:
            code.append(line)

    return '\n'.join(code)


def get_directives(repo_dir, repo_name):
    '''
    Extract all 'define' and 'typedef' directives from a given repo

    Paramenters:
        repo_name   - the name of the repo all the directives extracted from
    '''

    headers = get_headers(repo_dir, repo_name)
    defines =  set()
    typedefs = set()

    for header in headers:
        # open file and extract #define and typedef
        with open(os.path.join(repo_dir, repo_name, header), 'r')  as f:
            try:
                code = f.read()
            except UnicodeDecodeError:
                continue
            
            # remove comments
            LINE_COMMENT_RE = re.compile("//.*?\n" )
            MULTILINE_COMMENT_RE = re.compile("/\*.*?\*/", re.DOTALL)
            code = LINE_COMMENT_RE.sub("", code)
            code  = MULTILINE_COMMENT_RE.sub("", code)

            # join splitted lines in code
            code = join_splited_lines(code)

            for directive in re.findall(r'#(\s*)define(\s*)(\w+)(\s)', code):
                defines.add(f' {directive[2]} 1')

            for directive in re.findall(r'#(\s*)define(\s*)(\w+)\((.*?)\)(.*)\n', code):
                defines.add(f' {directive[2]} 1')
                # defines.add(f' {directive[2]}({directive[3]}) 1')

            for directive in re.findall(r'(\s*)typedef(\s*)(\w+)(\s*)(\w+)(\s*);', code):
                typedefs.add(f' int {directive[4]}')

            for directive in re.findall(r'(\s*)typedef(\s*)struct(\s*){(.+?)}(.+?);', code):
                typedefs.add(f' int {directive[4]}')

    return defines, typedefs


def extract_common_directives():
    '''
    Create fake headers that contain all defines and typedefs from /usr/include
    '''
    fake_dir = 'utils'

    if not os.path.exists(fake_dir):
        os.makedirs(fake_dir)

    with open(os.path.join(fake_dir, COMMON_FAKE_DEFINES), 'w+') as define_file, open(os.path.join(fake_dir, COMMON_FAKE_TYPEDEFS), 'w+') as typedef_file:
        defines, typedefs = get_directives('/usr', 'include')

        for define in defines:
            define_file.write(f'#define {define}\n')

        for typedef in typedefs:
            typedef_file.write(f'typedef {typedef};\n')

extract_common_directives()


def extract_all_directives():
    '''
    Create fake headers that contain all defines and typedefs
    '''
    fake_dir = 'utils'
    defines_set = set()
    typedefs_set = set()
    relevant_repos = os.listdir(REPOS_OMP_DIR)

    if not os.path.exists(fake_dir):
        os.makedirs(fake_dir)

    print('========== create fake headers ==========')
    # iterate over all the relevant repositories
    for idx, repo_name in enumerate(os.listdir(REPOS_DIR)):

        if repo_name not in relevant_repos:
            continue

        defines, typedefs = get_directives(repo_name)

        for define in defines:
            defines_set.add(define)

        for typedef in typedefs:
            typedefs_set.add(typedef)

        if idx > 0 and idx % 10**2 == 0:
            print(f'repos passed: {idx}')

    with open(os.path.join(fake_dir, FAKE_DEFINES), 'w+') as define_file, open(os.path.join(fake_dir, FAKE_TYPEDEFS), 'w+') as typedef_file:

        for define in defines_set:
            define_file.write(f'#define {define}\n')

        for typedef in typedefs_set:
            typedef_file.write(f'typedef {typedef};\n')




def extract_includes(file_path):
    '''
    Extract all #include directive from given file
    '''

    includes =  []

    with open(file_path, 'r')  as f:
        try:
            code = f.read()
        except UnicodeDecodeError:
            return []

        for directive in re.findall(r'#(\s*)include(\s*)("|<)(.*)("|>)', code):
            includes.append(directive[3])

    return includes


def create_fake_headers(directory, headers):
    '''
    Create all fake headers file and fill them with fake #include
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

    for header in headers:
        with open(os.path.join(directory, header), "w") as f:
            f.write(FAKE_INCLUDE)
   
def remove_fake_headers(directory, headers):
    '''
    Remove given files
    '''
    for header in headers:
        os.remove(os.path.join(directory, header))

# extract_all_directives()




# print(extract_includes("/home/talkad/Downloads/thesis/data_gathering_script/asd/123/cwt.c"))
# ['stdio.h', 'stdlib.h', 'math.h', 'time.h', 'sys/time.h']
# create_fake_headers('utils', ['stdio.h', 'stdlib.h', 'math.h', 'time.h', 'sys/time.h'])