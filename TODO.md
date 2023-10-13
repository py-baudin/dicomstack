# TODO

## optimize
- replace parsing and casting values by using DicomElement as a wrapper over pydicom.DataElement
  parse (and buffer) values only when asked
- parse DICOMDIR object?


## misc
- handle UID objects
  - auto-translate UID strings to names, using pydicom.uid.UID_dictionary
  - handle in queries: `stack.SOPClassUID == Raw Data Storage`
