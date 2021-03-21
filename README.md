# CoSyNE
Cooperative Synapse Neuro Evolution in Rust

### Demonstration

![cart_pole_champion](img/cart_pole_champion.gif)

### Features:
- User defined neural network topology using ANN struct
- User defined activation function through Config
    - Linear
    - Threshold
    - Sign
    - Sigmoid
    - Tanh
    - SoftSign
    - BentIdentity
    - Relu

### How to use
To use this crate in your project, add the following to your Cargo.toml:
```toml
[dependencies]
cosyne = { git = "https://github.com/MathisWellmann/cosyne" }
```

### Network Topology Creation
// TODO

### Example
// TODO

### Plot feature
// TODO:

## TODOS:
- add bit-wise fitness just like the paper suggests
- changing mutation probabilities and mutation strengths based on current generation
- multipoint crossover vs singlepoint crossover
- User defined initial network randomization method
    - uniform
    - gaussian
    - poisson disk sampling

### Donations :moneybag: :money_with_wings:
I you would like to support the development of this crate, feel free to send over a donation:

Monero (XMR) address:
```plain
47xMvxNKsCKMt2owkDuN1Bci2KMiqGrAFCQFSLijWLs49ua67222Wu3LZryyopDVPYgYmAnYkSZSz9ZW2buaDwdyKTWGwwb
```

![monero](img/monero_donations_qrcode.png)

### License
Copyright (C) 2020  <Mathis Wellmann wellmannmathis@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

![GNU AGPLv3](img/agplv3.png)
