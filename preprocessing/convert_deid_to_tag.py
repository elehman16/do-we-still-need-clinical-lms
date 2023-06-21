#{'URL', 'ORGANIZATION', 'AGE', 'NAME', 'PROFESSION', 'ID', 'MISC', 'EMAIL', 'LOCATION', 'CONTACT', 'DATE'}
TYPE_TO_TAG = {
                'URL': '[URL]',
                'ORGANIZATION': '[Company]',
                'AGE': '[Age]',
                'NAME': '[Name]',
                'PROFESSION': '[MISC]',
                'ID': '[Reg#]',
                'EMAIL': '[CI]',
                'LOCATION': '[LOC]',
                'CONTACT': '[CI]',
                'DATE': '[DR]',
                'MISC': '[MISC]'
              }

DEID_TO_TAG = {
               'First Name': '[Name]',
               'Last Name': '[Name]',
               'Clip Number (Radiology)': '[Reg#]',
               'Hospital': '[Company]',
               'Numeric Identifier': '[Reg#]',
               'Location': '[LOC]',
               'Initials': '[Name]',
               'Known lastname': '[Name]',
               'Known firstname': '[Name]',
               'Age over 90': '[Age]',
               'Medical Record Number': '[Reg#]',
               'Telephone/Fax': '[#]',
               'Name': '[Name]',
               'Street Address': '[LOC]',
               'Date Range': '[DR]',
               'Date range': '[DR]',
               'Serial Number': '[#]',
               'Social Security Number': '[#]',
               'State': '[State]',
               'E-mail address': '[CI]',
               'Company': '[Company]',
               'Pager number': '[#]',
               'Country': '[Country]',
               'MD Number': '[Reg#]',
               'Wardname': '[LOC]',
               'College': '[Company]',
               'Apartment Address': '[LOC]',
               'URL': '[URL]',
               'CC Contact Info': '[CI]',
               'Attending Info': '[CI]',
               'Job Number': '[Reg#]',
               'Year': '[YR]',
               'Month': '[MO]',
               'Day': '[DAY]',
               'Holiday': '[DAY]',
               'Unit Number': '[Reg#]',
               'PO Box': '[LOC]',
               'Dictator Info': '[CI]',
               'Provider Number': '[#]'
               }



def convert_deid_to_tag(t: str):
    for deid in DEID_TO_TAG.keys():
        if deid in t or deid.lower() in t:
            return DEID_TO_TAG[deid]
           
    if len(t.split('-')) == 3:
        return t
    
    elif t.strip().isnumeric():
        return '[Reg#]'

    elif t.replace('-', '').replace('/', '').isnumeric():
        return t

    elif 'January' in t:
        return t
    elif 'February' in t:
        return t
    elif 'March' in t:
        return t    
    elif 'April' in t:
        return t
    elif 'May' in t:
        return t
    elif 'June' in t:
        return t
    elif 'July' in t:
        return t
    elif 'August' in t:
        return t
    elif 'September' in t:
        return t
    elif 'October' in t:
        return t
    elif 'November' in t:
        return t
    elif 'December' in t:
        return t
    elif '' == t.strip():
        return ''
    else:
        raise NotImplementedError("Not available.")

def convert_tag_type_to_tag(type_: str) -> str:
    """Given a type of tag, return the proper tag. """
    return TYPE_TO_TAG[type_]
