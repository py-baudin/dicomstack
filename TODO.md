# TODO


## Add query-style `Field` object:

```
stack(Field.EchoTime == 1)
stack(Field.EchoTime > 2)
stack(Field.EchoTime in [1,2])
etc.

also: get values
stack[Field.EchoTime]

```

keep legacy query:
```
stack(EchoTime=1)
stack(EchoTime=[1,2])
stack["EchoTime"]
etc.
```
